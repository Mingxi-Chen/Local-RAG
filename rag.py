"""
本地RAG系统 - 基于Elasticsearch的PDF文档问答系统
支持文本、图片、表格的混合检索和精排
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
from config import ES_USERNAME, ES_PASSWORD, OPENAI_API_KEY, OPENAI_MODEL
from openai import OpenAI
# logging output to file
from logging.handlers import RotatingFileHandler
import os

log_file = os.path.join(os.path.dirname(__file__), "rag.log")
file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)


# PDF处理相关
import pdfplumber
import fitz  # PyMuPDF
import camelot
import pandas as pd

# NLP相关
import tiktoken
import numpy as np

# Elasticsearch
from elasticsearch import Elasticsearch, helpers, BadRequestError

# HTTP请求
import requests

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Update the Config class URLs
@dataclass
class Config:
    """系统配置"""
    # Elasticsearch配置
    ES_HOST: str = "localhost"
    ES_PORT: int = 9200
    ES_INDEX: str = "rag_documents"
    ES_USERNAME: str = ES_USERNAME  
    ES_PASSWORD: str = ES_PASSWORD 
    
    # API配置 - Updated to use external APIs
    EMBEDDING_URL: str = "http://test.2brain.cn:9800/v1/emb"
    RERANK_URL: str = "http://test.2brain.cn:2260/rerank"
    LLM_URL: str = "https://api.openai.com/v1/responses"
    IMAGE_MODEL_URL: str = "http://test.2brain.cn:23333/v1"

    # OpenAI配置
    OPENAI_API_KEY: str = OPENAI_API_KEY
    OPENAI_MODEL: str = OPENAI_MODEL
    
    # 文档处理参数
    CHUNK_SIZE: int = 800  # tokens
    CHUNK_OVERLAP: int = 100  # tokens
    
    # 检索参数
    BM25_WEIGHT: float = 0.3
    VECTOR_WEIGHT: float = 0.7
    TOP_K_RETRIEVE: int = 50
    TOP_K_RERANK: int = 5
    
    # 向量维度
    EMBEDDING_DIM: int = 1024

config = Config()

# Add missing imports at the top
import base64
import mimetypes
import time
import traceback
from openai import OpenAI

# ==================== 数据结构 ====================
@dataclass
class Chunk:
    """文档片段"""
    chunk_id: str
    doc_id: str
    page: int
    content: str
    content_type: str  # text, table, image
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class SearchResult:
    """搜索结果"""
    chunk_id: str
    content: str
    score: float
    doc_id: str
    page: int
    metadata: Dict[str, Any]
    content_type: str

# ==================== PDF解析模块 ====================
class PDFExtractor:
    """PDF内容提取器"""
    
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def extract_all(self, pdf_path: str) -> Dict[str, Any]:
        """提取PDF中的所有内容"""
        doc_id = self._generate_doc_id(pdf_path)
        result = {
            "doc_id": doc_id,
            "doc_path": pdf_path,
            "text_chunks": [],
            "tables": [],
            "images": []
        }
        
        # 提取文本
        logger.info(f"提取文本: {pdf_path}")
        text_content = self._extract_text(pdf_path)
        result["text_chunks"] = text_content
        
        # 提取表格
        logger.info(f"提取表格: {pdf_path}")
        tables = self._extract_tables(pdf_path)
        result["tables"] = tables
        
        # 提取图片
        logger.info(f"提取图片: {pdf_path}")
        images = self._extract_images(pdf_path)
        result["images"] = images
        
        return result
    
    def _generate_doc_id(self, pdf_path: str) -> str:
        """生成文档ID"""
        return hashlib.md5(pdf_path.encode()).hexdigest()[:16]
    
    def _extract_text(self, pdf_path: str) -> List[Dict]:
        """提取纯文本"""
        text_chunks = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    text_chunks.append({
                        "page": page_num,
                        "content": text,
                        "type": "text"
                    })
        
        return text_chunks
    
    def _extract_tables(self, pdf_path: str) -> List[Dict]:
        """提取表格"""
        tables = []
        
        try:
            # 使用camelot提取表格，添加错误处理参数
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # 忽略PDF结构警告
                table_list = camelot.read_pdf(
                    pdf_path, 
                    pages='all', 
                    flavor='stream',
                    suppress_stdout=True,  # 抑制标准输出
                    layout_kwargs={'detect_vertical': False}  # 减少检测复杂性
                )
            
            for idx, table in enumerate(table_list):
                try:
                    df = table.df
                    
                    # 检查表格是否有效（至少有数据）
                    if df.empty or df.shape[0] <= 1:
                        continue
                    
                    # 生成表格摘要
                    summary = self._generate_table_summary(df)
                    
                    tables.append({
                        "page": table.page,
                        "content": df.to_csv(index=False),
                        "summary": summary,
                        "type": "table",
                        "table_index": idx
                    })
                except Exception as table_error:
                    logger.debug(f"跳过无效表格 {idx}: {table_error}")
                    continue
                    
        except Exception as e:
            logger.warning(f"表格提取失败: {e}")
        
        return tables
    
    def _extract_images(self, pdf_path: str) -> List[Dict]:
        """提取图片"""
        images = []
        
        try:
            # 抑制fitz的警告信息
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pdf = fitz.open(pdf_path)
            
            for page_num, page in enumerate(pdf, 1):
                try:
                    image_list = page.get_images()
                    
                    for img_index, img in enumerate(image_list):
                        try:
                            # 提取图片数据
                            xref = img[0]
                            pix = fitz.Pixmap(pdf, xref)
                            
                            if pix.n - pix.alpha < 4:  # GRAY or RGB
                                img_data = pix.tobytes("png")
                                
                                # 调用图片摘要API
                                description = self._summarize_image(img_data)
                                
                                images.append({
                                    "page": page_num,
                                    "content": description,
                                    "type": "image",
                                    "image_index": img_index
                                })
                            
                            pix = None
                        except Exception as img_error:
                            logger.debug(f"跳过无效图片 {img_index} (页面{page_num}): {img_error}")
                            continue
                            
                except Exception as page_error:
                    logger.debug(f"页面{page_num}图片提取失败: {page_error}")
                    continue
            
            pdf.close()
        except Exception as e:
            logger.warning(f"图片提取失败: {e}")
        
        return images
    
    def _generate_table_summary(self, df: pd.DataFrame) -> str:
        """生成表格摘要"""
        rows, cols = df.shape
        # 确保列名都是字符串类型
        headers = ", ".join(str(col) for col in df.columns.tolist()[:5])  # 前5个列名
        
        summary = f"表格包含{rows}行{cols}列，主要列包括: {headers}"
        
        # 添加数据统计信息
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe().to_string()
            summary += f"\n数值统计信息:\n{stats[:200]}..."
        
        return summary
    
    def _summarize_image(self, img_data: bytes) -> str:
        """使用外部API生成图片描述"""
        try:
            # Save image temporarily to use with external API
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_file.write(img_data)
                tmp_file_path = tmp_file.name
            
            # Use external image summarization function
            description = self._call_external_image_api(tmp_file_path)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return description if description else "图片描述生成失败"
        except Exception as e:
            logger.error(f"图片摘要失败: {e}")
            return "图片描述生成异常"
    
    def _call_external_image_api(self, image_path: str) -> str:
        """调用外部图片摘要API"""
        retry = 0
        while retry <= 3:
            try:
                text = "详细地描述这张图片的内容，不要漏掉细节，并提取图片中的文字。注意只需客观说明图片内容，无需进行任何评价。"
                
                client = OpenAI(api_key=OPENAI_API_KEY, base_url=config.IMAGE_MODEL_URL)

                # Read local image and convert to Base64 data URL
                with open(image_path, 'rb') as f:
                    content_bytes = f.read()
                mime_type = mimetypes.guess_type(image_path)[0] or 'image/png'
                encoded = base64.b64encode(content_bytes).decode('utf-8')
                data_url = f"data:{mime_type};base64,{encoded}"
                
                resp = client.chat.completions.create(
                    model='internvl-internlm2',
                    messages=[{
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': text}, 
                            {'type': 'image_url','image_url': { 'url': data_url}}]
                        }], 
                    temperature=0.8, 
                    top_p=0.8, 
                    max_tokens=2048, 
                    stream=False,
                    timeout=60  # 添加超时
                )
                
                return resp.choices[0].message.content
            except Exception as e:
                logger.error(f"图片API调用失败 (重试{retry}): {e}")
                time.sleep(2)  # 增加等待时间
                retry += 1
            
        return "图片描述生成失败"

# ==================== 文档切分与向量化 ====================
class ChunkVectorizer:
    """文档切分和向量化"""
    
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def process_document(self, extracted_data: Dict) -> List[Chunk]:
        """处理提取的文档数据"""
        chunks = []
        doc_id = extracted_data["doc_id"]
        
        # 处理文本
        for text_data in extracted_data["text_chunks"]:
            text_chunks = self._chunk_text(
                text_data["content"],
                doc_id,
                text_data["page"]
            )
            chunks.extend(text_chunks)
        
        # 处理表格
        for table_data in extracted_data["tables"]:
            table_chunk = self._create_table_chunk(
                table_data,
                doc_id
            )
            chunks.append(table_chunk)
        
        # 处理图片
        for image_data in extracted_data["images"]:
            image_chunk = self._create_image_chunk(
                image_data,
                doc_id
            )
            chunks.append(image_chunk)
        
        # 批量生成向量
        chunks = self._batch_embed(chunks)
        
        return chunks
    
    def _chunk_text(self, text: str, doc_id: str, page: int) -> List[Chunk]:
        chunks = []
        tokens = self.tokenizer.encode(text)

        size = config.CHUNK_SIZE
        overlap = config.CHUNK_OVERLAP
        step = max(1, size - overlap)

        chunk_index = 0
        for start in range(0, len(tokens), step):
            end = min(start + size, len(tokens))
            chunk_tokens = tokens[start:end]
            if not chunk_tokens:
                break

            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_p{page}_c{chunk_index}",
                doc_id=doc_id,
                page=page,
                content=chunk_text,
                content_type="text",
                metadata={"chunk_index": chunk_index, "token_count": len(chunk_tokens)}
            ))
            if end == len(tokens):
                break
            chunk_index += 1

        return chunks
    
    def _create_table_chunk(self, table_data: Dict, doc_id: str) -> Chunk:
        """创建表格chunk"""
        chunk_id = f"{doc_id}_p{table_data['page']}_t{table_data['table_index']}"
        
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            page=table_data["page"],
            content=table_data["summary"],
            content_type="table",
            metadata={
                "table_index": table_data["table_index"],
                "csv_content": table_data["content"]
            }
        )
    
    def _create_image_chunk(self, image_data: Dict, doc_id: str) -> Chunk:
        """创建图片chunk"""
        chunk_id = f"{doc_id}_p{image_data['page']}_i{image_data['image_index']}"
        
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            page=image_data["page"],
            content=image_data["content"],
            content_type="image",
            metadata={
                "image_index": image_data["image_index"]
            }
        )
    
    def _batch_embed(self, chunks: List[Chunk]) -> List[Chunk]:
        """批量生成向量 - 使用外部embedding API"""
        texts = [chunk.content for chunk in chunks]
        
        # 添加详细调试信息
        print(f"    准备向量化 {len(texts)} 个文本块...")
        logger.info(f"准备向量化 {len(texts)} 个文本块")
        
        # 打印每个chunk的内容长度
        for i, (chunk, text) in enumerate(zip(chunks, texts)):
            content_length = len(text)
            print(f"    块{i+1}: 类型={chunk.content_type}, 长度={content_length}字符")
            if content_length > 1000:  # 如果内容很长，显示前100字符
                print(f"    内容预览: {text[:100]}...")
        
        try:
            # Use external embedding API format
            headers = {"Content-Type": "application/json"}
            data = {"texts": texts}
            
            print("    正在调用embedding API...")
            logger.info("开始调用embedding API")
            
            # 添加超时设置
            response = requests.post(
                config.EMBEDDING_URL, 
                headers=headers, 
                json=data,
                timeout=120  # 2分钟超时
            )
            
            print(f"    API响应状态: {response.status_code}")
            logger.info(f"embedding API响应状态: {response.status_code}")
            
            response.raise_for_status()
            
            if response.status_code == 200:
                result = response.json()
                embeddings = result['data']['text_vectors']
                
                print(f"    ✓ 成功获取 {len(embeddings)} 个向量")
                logger.info(f"成功获取 {len(embeddings)} 个向量")
                
                for chunk, embedding in zip(chunks, embeddings):
                    chunk.embedding = embedding
            else:
                print(f"    ✗ Embedding API错误: {response.status_code}")
                logger.error(f"Embedding API错误: {response.status_code}")
        except requests.exceptions.Timeout:
            print("    ✗ Embedding API调用超时")
            logger.error("Embedding API调用超时")
        except Exception as e:
            print(f"    ✗ Embedding失败: {e}")
            logger.error(f"Embedding失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return chunks

# ==================== Elasticsearch索引模块 ====================
class ESIndexer:
    """Elasticsearch索引管理"""
    
    def __init__(self):
        # 使用认证信息创建Elasticsearch连接
        self.es = Elasticsearch(
            [f"http://{config.ES_HOST}:{config.ES_PORT}"],
            basic_auth=(config.ES_USERNAME, config.ES_PASSWORD),  # 添加认证
            verify_certs=False
        )
    
    def _init_index(self):
        """初始化索引"""
        if not self.es.indices.exists(index=config.ES_INDEX):
            mapping = {
                "mappings": {
                    "properties": {
                        "chunk_id": {"type": "keyword"},
                        "doc_id": {"type": "keyword"},
                        "page": {"type": "integer"},
                        "content": {"type": "text", "analyzer": "ik_max_word"},
                        "content_type": {"type": "keyword"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": config.EMBEDDING_DIM,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "metadata": {"type": "object"},
                        "timestamp": {"type": "date"}
                    }
                }
            }
            self.es.indices.create(index=config.ES_INDEX, body=mapping)
            logger.info(f"创建索引: {config.ES_INDEX} (analyzer=ik_max_word)")
        """初始化索引"""
        if not self.es.indices.exists(index=config.ES_INDEX):
            def build_mapping(analyzer: str):
                return {
                    "mappings": {
                        "properties": {
                            "chunk_id": {"type": "keyword"},
                            "doc_id": {"type": "keyword"},
                            "page": {"type": "integer"},
                            "content": {"type": "text", "analyzer": analyzer},
                            "content_type": {"type": "keyword"},
                            "embedding": {
                                "type": "dense_vector",
                                "dims": config.EMBEDDING_DIM,
                                "index": True,
                                "similarity": "cosine"
                            },
                            "metadata": {"type": "object"},
                            "timestamp": {"type": "date"}
                        }
                    }
                }

            analyzers_to_try = ["ik_max_word", "standard"]
            last_error = None
            for analyzer in analyzers_to_try:
                try:
                    mapping = build_mapping(analyzer)
                    self.es.indices.create(index=config.ES_INDEX, body=mapping)
                    logger.info(f"创建索引: {config.ES_INDEX} (analyzer={analyzer})")
                    last_error = None
                    break
                except BadRequestError as e:
                    last_error = e
                    # IK 未安装时回退到 standard
                    if "analyzer [ik_max_word] has not been configured" in str(e):
                        logger.warning("未检测到 IK 分词器，回退使用 standard 分词器")
                        continue
                    raise
            if last_error:
                raise last_error
    def index_chunks(self, chunks: List[Chunk]):
        """批量索引文档"""
        # 确保索引存在
        self._init_index()
        
        actions = []
        
        for chunk in chunks:
            action = {
                "_index": config.ES_INDEX,
                "_id": chunk.chunk_id,
                "_source": {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "page": chunk.page,
                    "content": chunk.content,
                    "content_type": chunk.content_type,
                    "embedding": chunk.embedding,
                    "metadata": chunk.metadata,
                    "timestamp": datetime.now()
                }
            }
            actions.append(action)
        
        helpers.bulk(self.es, actions)
        logger.info(f"索引了 {len(chunks)} 个文档片段")
        logger.info(f"索引了 {len(chunks)} 个文档片段")

# ==================== 混合检索模块 ====================
class HybridSearcher:
    """混合检索器"""
    
    def __init__(self):
        # 使用认证信息创建Elasticsearch连接
        self.es = Elasticsearch(
            [f"http://{config.ES_HOST}:{config.ES_PORT}"],
            basic_auth=(config.ES_USERNAME, config.ES_PASSWORD),  # 添加认证
            verify_certs=False
        )
    
    def search(self, query: str, top_k: int = 50) -> List[SearchResult]:
        """混合检索"""
        # 获取查询向量
        query_embedding = self._get_query_embedding(query)
        
        # BM25检索
        bm25_results = self._bm25_search(query, top_k)
        
        # 向量检索
        vector_results = self._vector_search(query_embedding, top_k)
        
        # 结果融合
        merged_results = self._merge_results(bm25_results, vector_results)
        
        # Rerank
        reranked_results = self._rerank(query, merged_results[:config.TOP_K_RETRIEVE])
        
        return reranked_results[:config.TOP_K_RERANK]
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """获取查询向量 - 使用外部embedding API"""
        try:
            # Use external embedding API format
            headers = {"Content-Type": "application/json"}
            data = {"texts": [query]}
            
            response = requests.post(config.EMBEDDING_URL, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result['data']['text_vectors'][0]
            else:
                logger.error(f"查询向量API错误: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"查询向量生成失败: {e}")
            return None
    
    def _bm25_search(self, query: str, top_k: int) -> List[SearchResult]:
        """BM25检索"""
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content^2", "metadata.summary"],
                    "type": "best_fields"
                }
            },
            "size": top_k
        }
        
        response = self.es.search(index=config.ES_INDEX, body=body)
        
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            results.append(SearchResult(
                chunk_id=source["chunk_id"],
                content=source["content"],
                score=hit["_score"],
                doc_id=source["doc_id"],
                page=source["page"],
                metadata=source["metadata"],
                content_type=source["content_type"]
            ))
        
        return results
    
    def _vector_search(self, query_embedding: List[float], top_k: int) -> List[SearchResult]:
        """向量检索"""
        if not query_embedding:
            return []
        
        body = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": top_k * 2
            },
            "size": top_k
        }
        
        response = self.es.search(index=config.ES_INDEX, body=body)
        
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            results.append(SearchResult(
                chunk_id=source["chunk_id"],
                content=source["content"],
                score=hit["_score"],
                doc_id=source["doc_id"],
                page=source["page"],
                metadata=source["metadata"],
                content_type=source["content_type"]
            ))
        
        return results
    
    def _merge_results(self, bm25_results: List[SearchResult], 
                      vector_results: List[SearchResult]) -> List[SearchResult]:
        """RRF融合结果"""
        k = 60  # RRF参数
        
        # 计算RRF分数
        rrf_scores = {}
        
        for rank, result in enumerate(bm25_results):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + \
                                         config.BM25_WEIGHT / (k + rank + 1)
        
        for rank, result in enumerate(vector_results):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + \
                                         config.VECTOR_WEIGHT / (k + rank + 1)
        
        # 合并结果
        chunk_id_to_result = {}
        for result in bm25_results + vector_results:
            if result.chunk_id not in chunk_id_to_result:
                chunk_id_to_result[result.chunk_id] = result
        
        # 排序
        sorted_chunk_ids = sorted(rrf_scores.keys(), 
                                 key=lambda x: rrf_scores[x], 
                                 reverse=True)
        
        merged_results = []
        for chunk_id in sorted_chunk_ids:
            result = chunk_id_to_result[chunk_id]
            result.score = rrf_scores[chunk_id]
            merged_results.append(result)
        
        return merged_results
    
    def _rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """重排序 - 使用外部rerank API"""
        if not results:
            return []
        
        try:
            # Prepare documents for external rerank API
            result_doc = [{"text": r.content} for r in results]
            
            # Call external rerank API
            response = requests.post(
                config.RERANK_URL, 
                json={"query": query, "documents": [doc['text'] for doc in result_doc]}
            )
            
            if response.status_code == 200:
                res = response.json()
                if res and 'scores' in res and len(res['scores']) == len(result_doc):
                    # Update scores
                    for idx, result in enumerate(results):
                        result.score = res['scores'][idx]
                    
                    # Sort by rerank score in descending order
                    results.sort(key=lambda x: x.score, reverse=True)
            else:
                logger.error(f"Rerank API错误: {response.status_code}")
        except Exception as e:
            logger.error(f"Rerank失败: {e}")
        
        return results

# ==================== 回答生成模块 ====================
class AnswerGenerator:
    """回答生成器"""
    
    def generate_answer(self, query: str, search_results: List[SearchResult]) -> str:
        """生成带引用的回答"""
        if not search_results:
            return "抱歉，没有找到相关信息。"
        
        # 构建上下文
        context = self._build_context(search_results)
        
        # 构建prompt
        prompt = self._build_prompt(query, context)
        
        # 调用LLM
        answer = self._call_llm(prompt)
        
        # 添加引用
        answer_with_citations = self._add_citations(answer, search_results)
        
        return answer_with_citations
    
    def _build_context(self, search_results: List[SearchResult]) -> str:
        """构建上下文"""
        context_parts = []
        
        for idx, result in enumerate(search_results, 1):
            source = f"[{idx}] 来源: 文档{result.doc_id}, 第{result.page}页"
            content = result.content[:500]  # 截断过长内容
            context_parts.append(f"{source}\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """构建prompt"""
        prompt = f"""基于以下参考信息回答问题。在回答中使用[数字]格式引用信息来源。

参考信息:
{context}

问题: {query}

请提供准确、完整的回答，并在相关位置标注引用来源[数字]。如果信息不足，请明确说明。

回答:"""
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """调用LLM API"""
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                # 添加调试信息
                print(f"正在调用LLM API (尝试 {retry_count + 1}/{max_retries})...")
                logger.info(f"LLM API调用开始 - 尝试 {retry_count + 1}")
                
                # 修正OpenAI API调用格式
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config.OPENAI_API_KEY}"
                }
                
                # 使用正确的OpenAI API格式
                payload = {
                    "model": config.OPENAI_MODEL,
                    "input": prompt,
                    "max_output_tokens": 4000
                }
                
                response = requests.post("https://api.openai.com/v1/responses", headers=headers, json=payload)
                
                print(f"LLM API响应状态: {response.status_code}")
                logger.info(f"LLM API响应状态: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    print("LLM返回:", json.dumps(result, ensure_ascii=False, indent=2))
                    
                    # 解析新的响应格式
                    answer = ""
                    
                    # 新版API格式: output是一个列表
                    if "output" in result and isinstance(result["output"], list):
                        for output_item in result["output"]:
                            # 找到type为"message"的输出
                            if output_item.get("type") == "message":
                                # content也是一个列表
                                content_list = output_item.get("content", [])
                                for content_item in content_list:
                                    # 找到type为"output_text"的内容
                                    if content_item.get("type") == "output_text":
                                        answer = content_item.get("text", "")
                                        break
                                if answer:
                                    break
                    
                    # 如果没找到答案，尝试其他可能的字段
                    if not answer:
                        # 尝试旧格式
                        answer = result.get("output_text", "")
                    
                    if answer:
                        print("✔ LLM API调用成功")
                        logger.info(f"LLM API调用成功，答案长度: {len(answer)}")
                        return answer.strip()
                    else:
                        # 如果响应完成但没有文本
                        if result.get("status") == "completed":
                            logger.error("LLM响应完成但未找到文本内容")
                            print("✗ LLM响应完成但未找到文本内容")
                        elif result.get("status") == "incomplete":
                            incomplete_reason = result.get("incomplete_details", {}).get("reason", "unknown")
                            logger.warning(f"LLM响应不完整: {incomplete_reason}")
                            print(f"⚠ LLM响应不完整: {incomplete_reason}")
                        
                        # 打印响应结构帮助调试
                        logger.error(f"无法从响应中提取答案，响应结构: {json.dumps(result, ensure_ascii=False)[:500]}")
                    
                else:
                    error_msg = f"LLM API错误 - 状态码: {response.status_code}"
                    try:
                        error_detail = response.json()
                        error_msg += f", 详情: {error_detail}"
                    except:
                        error_msg += f", 响应: {response.text[:200]}"
                    
                    logger.warning(error_msg)
                    print(f"✗ {error_msg}")
                    
            except requests.exceptions.Timeout as e:
                error_msg = f"LLM API调用超时 (重试{retry_count+1}/{max_retries}): {e}"
                logger.warning(error_msg)
                print(f"✗ {error_msg}")
                
            except requests.exceptions.ConnectionError as e:
                error_msg = f"LLM服务连接失败 (重试{retry_count+1}/{max_retries}): {e}"
                logger.warning(error_msg)
                print(f"✗ {error_msg}")
                
            except requests.exceptions.RequestException as e:
                error_msg = f"LLM API请求异常 (重试{retry_count+1}/{max_retries}): {e}"
                logger.warning(error_msg)
                print(f"✗ {error_msg}")
                
            except Exception as e:
                error_msg = f"LLM调用未知异常: {e}"
                logger.error(error_msg)
                print(f"✗ {error_msg}")
                import traceback
                logger.error(traceback.format_exc())
                break
            
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count  # 指数退避
                print(f"等待 {wait_time} 秒后重试...")
                import time
                time.sleep(wait_time)
        
        # 如果所有重试都失败，提供降级服务
        print("✗ LLM服务不可用，提供降级回答")
        logger.error("LLM服务不可用，提供基础回答")
        return self._fallback_answer(prompt)
    
    def _fallback_answer(self, prompt: str) -> str:
        """当LLM服务不可用时的降级回答"""
        return """抱歉，LLM服务暂时不可用。以下是可能的解决方案：

1. **检查API配置**：
   - 确认 OPENAI_API_KEY 是否有效
   - 检查 OPENAI_MODEL 配置是否正确
   - 验证网络连接是否正常

2. **服务状态检查**：
   - OpenAI API 服务是否正常
   - 是否存在API配额限制

3. **临时解决方案**：
   - 可以查看下方的检索结果获取相关信息
   - 稍后重试问答功能

请检查日志文件获取更详细的错误信息。"""
    
    def _add_citations(self, answer: str, search_results: List[SearchResult]) -> str:
        """添加详细引用"""
        citations = ["\n\n引用来源:"]
        
        for idx, result in enumerate(search_results, 1):
            citation = f"[{idx}] 文档: {result.doc_id}, 页码: {result.page}"
            if result.content_type == "table":
                citation += f" (表格{result.metadata.get('table_index', '')})"
            elif result.content_type == "image":
                citation += f" (图片{result.metadata.get('image_index', '')})"
            citations.append(citation)
        
        return answer + "\n".join(citations)

# ==================== 主流程控制 ====================
class RAGPipeline:
    """RAG系统主流程"""
    
    def __init__(self):
        self.extractor = PDFExtractor()
        self.vectorizer = ChunkVectorizer()
        self.indexer = ESIndexer()
        self.searcher = HybridSearcher()
        self.generator = AnswerGenerator()
    
    def index_pdf(self, pdf_path: str):
        """索引PDF文档"""
        logger.info(f"开始处理PDF: {pdf_path}")
        
        # 1. 提取内容
        extracted_data = self.extractor.extract_all(pdf_path)
        
        # 2. 切分和向量化
        chunks = self.vectorizer.process_document(extracted_data)
        
        # 3. 索引到ES
        self.indexer.index_chunks(chunks)
        
        logger.info(f"PDF处理完成: {pdf_path}")
        return extracted_data["doc_id"]
    
    def query(self, question: str) -> str:
        """查询问答"""
        print(f"\n开始处理问题: {question}")
        logger.info(f"查询: {question}")
        
        try:
            # 1. 混合检索
            print("1. 开始混合检索...")
            search_results = self.searcher.search(question, config.TOP_K_RETRIEVE)
            print(f"✓ 检索完成，找到 {len(search_results)} 个相关结果")
            logger.info(f"检索到 {len(search_results)} 个结果")
            
            # 2. 生成回答
            print("2. 开始生成回答...")
            answer = self.generator.generate_answer(question, search_results)
            print("✓ 回答生成完成")
            logger.info("回答生成完成")
            
            return answer
            
        except Exception as e:
            error_msg = f"查询处理异常: {e}"
            print(f"✗ {error_msg}")
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            return f"查询处理失败: {error_msg}"
    
    def batch_index_pdfs(self, pdf_paths: List[str]):
        """批量索引PDF"""
        doc_ids = []
        for pdf_path in pdf_paths:
            try:
                doc_id = self.index_pdf(pdf_path)
                doc_ids.append(doc_id)
            except Exception as e:
                logger.error(f"处理PDF失败 {pdf_path}: {e}")
        
        return doc_ids

# ==================== 使用示例 ====================
def main():
    """主函数示例"""
    # 初始化RAG系统
    rag = RAGPipeline()
    
    # 索引PDF文档
    pdf_files = [
        "test_pdf/image_extraction_example.pdf"
    ]
    
    # 批量索引
    doc_ids = rag.batch_index_pdfs(pdf_files)
    print(f"已索引文档: {doc_ids}")
    
    # 查询示例
    questions = [
        "什么是对话系统(Dialog Systems)?"
        
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        answer = rag.query(question)
        print(f"回答: {answer}")
        print("-" * 50)

# ==================== API服务（可选） ====================
from flask import Flask, request, jsonify

app = Flask(__name__)
rag_pipeline = None

@app.route('/index', methods=['POST'])
def index_endpoint():
    """索引PDF接口"""
    data = request.json
    pdf_path = data.get('pdf_path')
    
    if not pdf_path:
        return jsonify({"error": "Missing pdf_path"}), 400
    
    try:
        doc_id = rag_pipeline.index_pdf(pdf_path)
        return jsonify({
            "status": "success",
            "doc_id": doc_id
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/query', methods=['POST'])
def query_endpoint():
    """查询接口"""
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "Missing question"}), 400
    
    try:
        answer = rag_pipeline.query(question)
        return jsonify({
            "status": "success",
            "answer": answer
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

def start_api_server():
    """启动API服务"""
    global rag_pipeline
    rag_pipeline = RAGPipeline()
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    # 运行示例
    main()
    
    # 或启动API服务
    # start_api_server()