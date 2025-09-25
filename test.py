#!/usr/bin/env python3
"""
RAG系统测试脚本
"""
import os
import sys
import time
from rag import RAGPipeline, config

def test_system():
    """测试RAG系统基本功能"""
    print("=== RAG系统测试开始 ===")
    
    # 1. 测试系统初始化
    print("1. 初始化RAG系统...")
    try:
        rag = RAGPipeline()
        print("✓ RAG系统初始化成功")
    except Exception as e:
        print(f"✗ RAG系统初始化失败: {e}")
        return False
    
    # 3. 测试PDF文档索引 - 分别测试各个组件
    test_pdf_paths = get_test_pdf_paths()
    print(f"\n3. 测试PDF文档索引组件...")
    
    for pdf_path in test_pdf_paths:
        if os.path.exists(pdf_path):
            print(f"\n处理PDF: {pdf_path}")
            
            # 3.1 测试提取器
            print("\n  3.1 测试PDF提取器...")
            start_time = time.time()
            extracted_data = test_extractor(rag.extractor, pdf_path)
            print(f"    提取耗时: {time.time() - start_time:.2f}秒")
            
            # 3.2 测试向量化器 - 使用已提取的数据
            if extracted_data:
                print("\n  3.2 测试向量化器...")
                start_time = time.time()
                test_vectorizer_direct(rag.vectorizer, extracted_data)
                print(f"    向量化耗时: {time.time() - start_time:.2f}秒")

                # 3.3 测试索引器
                print("\n  3.3 测试索引器...")
                start_time = time.time()
                test_indexer(rag.vectorizer, rag.indexer, extracted_data)
                print(f"    索引耗时: {time.time() - start_time:.2f}秒")
                
        else:
            print(f"⚠ PDF文件不存在: {pdf_path}")
    
    if not test_pdf_paths:
        print("未配置测试PDF路径")

def test_extractor(extractor, pdf_path):
    """测试PDF提取器"""
    try:
        extracted_data = extractor.extract_all(pdf_path)
        # 检查返回的数据结构
        required_keys = ["doc_id", "doc_path", "text_chunks", "tables", "images"]
        for key in required_keys:
            if key not in extracted_data:
                raise ValueError(f"缺少必要字段: {key}")
        
        # 统计提取的内容
        text_count = len(extracted_data["text_chunks"])
        table_count = len(extracted_data["tables"])
        image_count = len(extracted_data["images"])
        
        print(f"    ✓ 提取完成 - 文档ID: {extracted_data['doc_id']}")
        print(f"    ✓ 文本块: {text_count}, 表格: {table_count}, 图片: {image_count}")
        
        return extracted_data
        
    except Exception as e:
        print(f"    ✗ PDF提取失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_vectorizer_direct(vectorizer, extracted_data):
    """直接测试向量化器 - 使用已提取的数据"""
    try:
        # 显示提取的数据统计
        text_count = len(extracted_data["text_chunks"])
        table_count = len(extracted_data["tables"])
        image_count = len(extracted_data["images"])
        print(f"    提取统计: 文本块{text_count}, 表格{table_count}, 图片{image_count}")
        
        # 显示图片内容长度（如果有的话）
        if image_count > 0:
            print("    图片描述长度:")
            for i, img in enumerate(extracted_data["images"]):
                print(f"      图片{i+1}: {len(img['content'])}字符")
                if len(img['content']) > 200:
                    print(f"      内容预览: {img['content'][:200]}...")
        
        # 处理文档
        print("    正在处理文档和生成向量...")
        chunks = vectorizer.process_document(extracted_data)
        
        if not chunks:
            print("    ⚠ 未生成任何文档块")
            return extracted_data
            
        # 检查生成的chunks
        chunk_count = len(chunks)
        embedded_count = sum(1 for chunk in chunks if chunk.embedding is not None)
        
        print(f"    ✓ 文档处理完成 - 生成块数: {chunk_count}")
        print(f"    ✓ 向量化完成 - 嵌入向量: {embedded_count}/{chunk_count}")
        
        # 检查chunk结构
        if chunks:
            sample_chunk = chunks[0]
            print(f"    ✓ 样本块 - ID: {sample_chunk.chunk_id[:16]}..., 类型: {sample_chunk.content_type}")
            if sample_chunk.embedding:
                print(f"    ✓ 向量维度: {len(sample_chunk.embedding)}")
        
        return extracted_data
        
    except Exception as e:
        print(f"    ✗ 向量化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_indexer(vectorizer, indexer, extracted_data):
    """测试索引器"""
    print("\n  3.3 测试索引器 (indexer.index_chunks)...")
    try:
        # 处理文档生成chunks
        chunks = vectorizer.process_document(extracted_data)
        if not chunks:
            print("    ✗ 无法获取文档块")
            return
            
        # 索引chunks
        indexer.index_chunks(chunks)
        
        print(f"    ✓ 索引完成 - 已索引 {len(chunks)} 个文档块")
        print(f"    ✓ 文档ID: {extracted_data['doc_id']} 索引到Elasticsearch")
        
    except Exception as e:
        print(f"    ✗ 索引失败: {e}")
        import traceback
        traceback.print_exc()

def test_apis():
    """测试外部API连接"""
    import requests
    
    # 测试embedding API
    try:
        response = requests.get(config.EMBEDDING_URL.replace('/v1/emb', '/health'), timeout=5)
        print("✓ Embedding API连接正常" if response.status_code < 500 else "⚠ Embedding API响应异常")
    except:
        print("✗ Embedding API连接失败")
    
    # 测试rerank API
    try:
        response = requests.get(config.RERANK_URL.replace('/rerank', '/health'), timeout=5)
        print("✓ Rerank API连接正常" if response.status_code < 500 else "⚠ Rerank API响应异常")
    except:
        print("✗ Rerank API连接失败")

def get_test_pdf_paths():
    """获取测试PDF文件路径"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_pdf_dir = os.path.join(base_dir, "test_pdf")
    pdf_paths = [
        os.path.join(test_pdf_dir, "image_extraction_example.pdf")
    ]
    return pdf_paths

if __name__ == "__main__":
    test_system()