FROM --platform=arm64 public.ecr.aws/lambda/python:3.9

# 复制依赖文件
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# 安装依赖
RUN pip install -r requirements.txt

RUN mkdir -p /tmp/matplotlib && \
    chmod 777 /tmp/matplotlib

ENV MPLCONFIGDIR=/tmp/matplotlib

# 复制函数代码
COPY lambda_function.py ${LAMBDA_TASK_ROOT}



# 设置处理函数
CMD [ "lambda_function.lambda_handler" ]

