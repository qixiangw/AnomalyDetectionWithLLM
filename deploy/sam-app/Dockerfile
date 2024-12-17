FROM public.ecr.aws/lambda/python:3.9

# 复制依赖文件
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# 安装依赖
RUN pip install -r requirements.txt

# 复制函数代码
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# 设置处理函数
CMD [ "lambda_function.lambda_handler" ]

