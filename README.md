一个基于NER和RE的菜谱收录系统。

  .env:环境变量配置文件。存储数据库URL、模型路径、训练参数（学习率、batch size、epochs等）和模型基础名称（bert-base-chinese）
  config.py:配置常量定义。读取 .env 环境变量，定义实体标签（食材/调料/份量/处理方式/动作/时间）、关系标签等全局常量
  requirements.py:Python依赖清单。
  database.py：SQLAlchemy数据库模型。定义4张表： Recipe （菜谱主表）、 Ingredient （食材表）、 Seasoning （调料表）、 Action （烹饪动作表），提供 init_db() 初始化和 get_db() 会话获取.
  recipes.db:SQLite数据库文件。实际存储菜谱数据的本地数据库
  data_preprocessor.py:数据预处理模块。包含 load_data() 加载数据、 convert_to_ner_format() 转换为BIO标注格式、 convert_to_re_format() 转换为关系分类格式（含负采样优化）、 get_ner_features() 和 get_re_features() 生成模型输入特征
  train_ner.py:NER（命名实体识别）模型训练脚本。加载BERT中文模型→预处理数据→Token分类微调→计算precision/recall/F1→保存模型。支持断点续训、cosine学习率调度、warmup预热
  train_re.py:RE（关系抽取）模型训练脚本。类似NER流程，使用序列分类模型对实体对进行关系分类。支持从checkpoint-599恢复训练，应用学习率预热和cosine衰减
  test_models.py:模型测试脚本。加载训练好的NER和RE模型，对示例文本进行实体识别和关系抽取验证
  inference.py:模型推理模块。 RecipeInference 类封装NER和RE模型，提供 predict_entities() 实体预测、 predict_relations() 关系预测、 extract_recipe() 完整抽取接口
  main.py:FastAPI服务主程序。提供RESTful API接口： /extract （抽取信息）、 /recipe （创建菜谱，含去重机制）、 /recipes （列表查询）、 /recipes/{id} （详情/删除）。配置CORS跨域支持，启动时自动加载模型
  recipe_re_dataset.zip:训练数据压缩包。解压后包含 train.json （960条）、 dev.json （120条）、 test.json （120条），已标注实体和关系


  数据流图：
  用户输入菜谱文本
       ↓
frontend/index.html  ──HTTP请求──→  main.py (/extract 或 /recipe)
                                      ↓
                                  inference.py
                                  ├── predict_entities() → 调用NER模型
                                  └── predict_relations() → 调用RE模型
                                      ↓
                                  database.py  ←──→  recipes.db
                                  （存储到Recipe/Ingredient/Seasoning/Action表）
                                      ↓
前端展示结果 ←──JSON响应────────── main.py


点击main.py运行后端程序后，点击index.html打开前端页面，进行菜谱录入、查询、删除等功能。模型存储位置：https://huggingface.co/YuZhaomu/RecipeExtract/tree/main ；
模型仍然需要优化：训练后发现模型对于食材和调料的区分度低，可能需要重新定义实体类型并进行数据标注、重新训练。
