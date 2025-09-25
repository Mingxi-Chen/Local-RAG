# 本地RAG系统

基于Elasticsearch的PDF文档问答系统，支持文本、图片、表格的混合检索和精排。

## 功能特性

- 📄 **多格式PDF处理**: 支持文本、图片、表格的提取和索引
- 🔍 **混合检索**: 结合向量检索和BM25文本检索
- 🎯 **智能重排**: 基于语义相似度的结果重排
- 🖼️ **图像理解**: 支持PDF中图像内容的理解和检索
- 📊 **表格解析**: 自动识别和解析PDF中的表格数据
- 🚀 **RESTful API**: 提供HTTP API接口，支持Web应用集成
- 📝 **批量处理**: 支持批量PDF文档索引和处理

## 系统架构

```
rag-development/
├── rag.py              # 主程序文件
├── config.py           # 配置文件
├── test.py             # 测试脚本
├── requirements.txt    # 依赖包列表
├── test_pdf/          # 测试PDF文件目录
├── rag.log            # 系统日志文件
└── README.md          # 项目说明文档
```

## 核心组件

### 1. PDF处理器
- **文本提取**: 使用 `pdfplumber` 提取文本内容
- **图像提取**: 使用 `PyMuPDF` 提取图像
- **表格解析**: 使用 `camelot` 解析表格数据

### 2. 向量化引擎
- 支持文本和图像的向量化
- 集成外部嵌入服务
- 自动处理不同类型内容

### 3. 检索系统
- **混合检索**: 结合向量检索和BM25
- **语义重排**: 基于相似度的结果重排
- **多模态支持**: 文本、图像、表格统一检索

### 4. 答案生成
- 集成OpenAI API
- 基于检索结果生成准确答案
- 支持上下文理解

## 环境要求

### 系统要求
- Python 3.8+
- Elasticsearch 7.0+
- 至少8GB内存（推荐16GB+）

### 外部服务依赖
- Elasticsearch集群
- OpenAI API（用于答案生成）
- 外部向量化服务（用于文本和图像向量化）

## 安装步骤

### 1. 克隆项目
```bash
git clone <repository-url>
cd local-RAG
```

### 2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 启动Elasticsearch
确保Elasticsearch服务正在运行：
```bash
# 使用Docker启动Elasticsearch
docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.17.0

# 或使用本地安装的Elasticsearch
elasticsearch
```

### 5. 配置服务
编辑 `config.py` 文件，设置：
- Elasticsearch连接信息
- OpenAI API密钥
- 外部向量化服务地址

## 运行程序

### 方式一：命令行运行
```bash
# 启动RAG系统
python rag.py

# 运行测试
python test.py
```

### 方式二：API服务模式
```bash
# 启动Flask API服务（默认端口5000）
python rag.py --api
```

然后可以通过HTTP请求使用：
```bash
# 索引PDF文档
curl -X POST http://localhost:5000/index \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "path/to/your/document.pdf"}'

# 查询文档
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "你的问题"}'
```

### 方式三：代码集成
```python
from rag import RAGSystem

# 初始化RAG系统
rag = RAGSystem()

# 索引PDF文档
rag.index_pdf("path/to/document.pdf")

# 查询文档
answer = rag.query("你的问题")
print(answer)
```

## 配置说明

### Elasticsearch配置
```python
# 在 rag.py 的 Config 类中
es_host = "localhost"
es_port = 9200
index_name = "rag_documents"
```

### 日志配置
- 日志文件: `rag.log`
- 日志级别: INFO
- 自动轮转: 5MB, 保留3个备份

## 故障排除

### 常见问题

**1. Elasticsearch连接失败**
```
解决方案:
- 检查Elasticsearch是否正在运行
- 验证用户名和密码是否正确
- 确认网络连接正常
```

**2. PDF处理失败**
```
解决方案:
- 检查PDF文件是否损坏
- 确认文件路径正确
- 验证文件权限
```

**3. API服务无法启动**
```
解决方案:
- 检查端口5000是否被占用
- 验证所有依赖是否正确安装
- 查看日志文件获取详细错误信息
```

**4. 向量化服务连接失败**
```
解决方案:
- 检查config.py中的服务地址配置
- 验证外部服务是否正常运行
- 确认网络连接和防火墙设置
```

### 日志查看
```bash
tail -f rag.log
```

## 性能优化

### 建议配置
- **内存**: 建议16GB+用于大规模文档处理
- **存储**: 使用SSD提升Elasticsearch性能
- **网络**: 确保与外部服务的网络延迟较低

### 批量处理优化
- 使用 `batch_index_pdfs()` 进行批量索引
- 合理设置批次大小避免内存溢出
- 监控Elasticsearch集群状态

## 扩展开发

### 添加新的文档类型
1. 在 `extract_content_from_pdf()` 中添加处理逻辑
2. 更新向量化流程
3. 测试新格式的处理效果

### 集成新的LLM
1. 修改 `Config` 类中的模型配置
2. 更新 `generate_answer()` 方法
3. 调整提示词模板

## 许可证

[根据实际情况添加许可证信息]

## 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 联系方式

[根据实际情况添加联系方式]
```

这个README文件包含了：

1. **项目概述**: 清晰描述项目功能和特性
2. **环境要求**: 详细的系统要求和依赖服务
3. **安装步骤**: 从零开始的完整安装指南
4. **使用方法**: 三种不同的使用方式（命令行、API、代码集成）
5. **项目结构**: 清晰的文件组织说明
6. **核心组件**: 系统架构的详细说明
7. **配置说明**: 重要配置项的解释
8. **故障排除**: 常见问题和解决方案
9. **性能优化**: 生产环境的建议
10. **扩展开发**: 为开发者提供的扩展指南

你可以根据实际情况调整其中的配置信息、联系方式等内容。需要我修改或补充任何部分吗？
