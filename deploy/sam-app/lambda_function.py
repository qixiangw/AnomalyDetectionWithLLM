import pandas as pd # type: ignore
import os
import base64
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt # type: ignore
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import boto3 # type: ignore
import json
import io
import re

# Global constants
MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
REGION = "us-west-2"  # AWS
TMP_DIR = "/tmp"  # Lambda临时目录
TIME_WINDOW_HOURS = 8 

# Global AWS clients
s3_client = boto3.client('s3', region_name=REGION)
bedrock_client = boto3.client('bedrock-runtime', region_name=REGION)

def read_cpu_data_from_s3(bucket: str, key: str) -> pd.DataFrame:
    """
    Read CPU usage data from CSV file in S3 with error handling and validation
    """
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(response['Body'].read()))
        metric_name = 'cpuusage'
        required_columns = ['service', 'pod', 'node', 'timestamp', metric_name]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    except Exception as e:
        raise Exception(f"Error reading CSV from S3: {str(e)}")

def group_cpu_data(df: pd.DataFrame) -> Dict[Tuple[str, str, str], pd.DataFrame]:
    """
    Group CPU usage data by service, pod, and node
    Returns a dictionary with (service, pod, node) as key and dataframe of timestamps and CPU usage as value
    """
    grouped_data = {}
    for (service, pod, node), group in df.groupby(['service', 'pod', 'node']):
        grouped_data[(service, pod, node)] = group[['timestamp', 'cpuusage']].sort_values('timestamp')
    return grouped_data

def calculate_statistics(grouped_data: Dict[Tuple[str, str, str], pd.DataFrame]) -> Dict[Tuple[str, str, str], Dict]:
    """
    Calculate basic statistics for each group
    """
    stats = {}
    for key, data in grouped_data.items():
        stats[key] = {
            'mean': data['cpuusage'].mean(),
            'max': data['cpuusage'].max(),
            'min': data['cpuusage'].min(),
            'std': data['cpuusage'].std()
        }
    return stats

def plot_cpu_usage(grouped_data: Dict[Tuple[str, str, str], pd.DataFrame]) -> Dict[str, List[Tuple[str, str, pd.DataFrame]]]:
    """
    Plot CPU usage for each service and save to temporary directory
    Returns: Dictionary with service as key and list of (pod, node, data) tuples as value
    """
    service_groups = {}
    
    for (service, pod, node), data in grouped_data.items():
        if service not in service_groups:
            service_groups[service] = []
        service_groups[service].append((pod, node, data))
    
    # Process each service
    processed_service_groups = {}
    for service, pod_data_list in service_groups.items():
        # Find the common time range for all pods in this service
        service_min_timestamp = min(data['timestamp'].min() for _, _, data in pod_data_list)
        service_max_timestamp = max(data['timestamp'].max() for _, _, data in pod_data_list)
        total_hours = (service_max_timestamp - service_min_timestamp).total_seconds() / 3600
        
        # Determine the window for all pods in this service
        if total_hours > TIME_WINDOW_HOURS:
            # Use a fixed random seed for reproducibility
            random.seed(42)
            # Same window start for all pods in this service
            window_start = service_min_timestamp + timedelta(hours=random.uniform(0, total_hours - TIME_WINDOW_HOURS))
            window_end = window_start + timedelta(hours=TIME_WINDOW_HOURS)
        else:
            window_start = service_min_timestamp
            window_end = service_max_timestamp
        
        processed_service_groups[service] = []
        
        # Process each pod's data using the same time window
        plt.figure(figsize=(15, 8))
        for pod, node, data in pod_data_list:
            # Filter data for the time window
            mask = (data['timestamp'] >= window_start) & (data['timestamp'] <= window_end)
            window_data = data[mask].copy()
            
            # Downsample by taking every other point
            downsampled_data = window_data.iloc[::2].copy()
            
            # Store the downsampled data
            processed_service_groups[service].append((pod, node, downsampled_data))
            
            # Plot the data
            label = f"{pod}-{node}"
            plt.plot(downsampled_data['timestamp'], downsampled_data['cpuusage'], label=label, marker='o', markersize=4)
        
        plt.xlabel('Time')
        plt.ylabel('CPU Usage (%)')
        plt.title(f'CPU Usage Over Time - {service}\n({TIME_WINDOW_HOURS}-hour window with downsampled data)')
        plt.grid(True)
        
        # Set x-axis ticks to show reasonable time intervals
        time_interval = timedelta(minutes=15)  # Show ticks every 15 minutes
        ticks = pd.date_range(start=window_start, end=window_end, freq=time_interval)
        plt.xticks(ticks, [t.strftime('%Y-%m-%d %H:%M') for t in ticks], rotation=45)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save plot to temporary directory
        tmp_path = os.path.join(TMP_DIR, f"cpu_usage_{service}.jpg")
        plt.savefig(tmp_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved plot for {service} to temp directory: {tmp_path}")
    
    return service_groups

def extract_result_content(xml_response: str) -> str:
    """
    Extract content from <result> tags in XML response
    """
    match = re.search(r'<result>(.*?)</result>', xml_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_pod_names(xml_response: str) -> List[str]:
    """
    Extract pod names from XML response when anomalies are detected
    """
    pod_names = []
    matches = re.finditer(r'<pod\d+>(.*?)</pod\d+>', xml_response, re.DOTALL)
    for match in matches:
        pod_names.append(match.group(1).strip())
    return pod_names

def llm_analysis(service_data: List[Tuple[str, str, pd.DataFrame]], service_name: str) -> str:
    """
    Analyze CPU usage data for a specific service using Bedrock
    """
    stats_text = "Statistics:\n"
    for pod, node, data in service_data:
        stats = {
            'mean': data['cpuusage'].mean(),
            'max': data['cpuusage'].max(),
            'min': data['cpuusage'].min(),
            'std': data['cpuusage'].std()
        }
        stats_text += f"Pod: {pod}\nNode: {node}\n"
        stats_text += f"Mean: {stats['mean']:.2f}, Max: {stats['max']:.2f}, Min: {stats['min']:.2f}, Std: {stats['std']:.2f}\n\n"
    
    raw_data_text = "Raw Data:\n"
    for pod, node, data in service_data:
        raw_data_text += f"Pod: {pod}\nNode: {node}\n"
        raw_data_text += f"Data:\n{data['cpuusage'].to_string()}\n\n"
    
    prompt_template = """你是一位拥有丰富AWS云平台运维经验的DevOps专家，特别擅长容器以及微服务中的日志和性能分析。请仔细分析所提供的EKS集群中同一Service下不同Pod的CPU使用率日志<log>，重点关注：
        1. 对比不同Pod之间的负载差异，如果有Pod与其他Pod有明显不同的CPU使用模式则视作异常Pod,注意缺乏正常负载波动特征的Pod如死平，长时间高负载等情况也视作异常Pod；
        2. 参考<data>中监控指标的统计数值，有均值，最值和方差，分析这些统计指标来辅助识别异常的Pod，比如某个pod的平均值偏离多数pod的平均值就是异常Pod；
        3. 忽略CPU使用率瞬间异常峰值或谷值，这有对应的监控告警。
        4. 返回异常Pod名称，如果没有异常的pod则返回没有。不要解释。
        请返回有异常pod名称格式如<example>，无需返回原因，解决方案或建议等多余的内容。
        输出格式参考：
        <example1>
            <result>
            无异常
            </result>
        </example1>
        <example2>
            <result>
                <pod1>
                a-b-f44568697-g8rz8-172.24.154.112
                </pod1>
                <pod2>
                a-b-f44568697-g8rz8-172.24.154.112
                </pod2>
            </result>
        </example2>
        <data>
        {stats}
        </data>
        <log>
        {data}
        </log>
        """.format(stats=stats_text, data=raw_data_text)
    
    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {"role": "user",
             "content": [
                 {"type": "text",
                  "text": prompt_template
                  }
                ]
            }
        ]
    }
    
    body = json.dumps(prompt_config)
    contentType = "application/json"
    accept = "application/json"
    
    response = bedrock_client.invoke_model(
        modelId=MODEL_ID,
        contentType=contentType,
        accept=accept,
        body=body
    )
    response_body = json.loads(response.get("body").read())
    return response_body.get("content")[0]["text"]

def llm_analysis_img(service_name: str) -> str:
    """
    Analyze CPU usage anomalies from a monitoring plot image using Claude
    """
    # Read image from temporary directory
    image_path = os.path.join(TMP_DIR, f"cpu_usage_{service_name}.jpg")
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    prompt_template = """
        你是一位拥有丰富AWS云平台运维经验的DevOps专家，特别擅长容器以及微服务中的日志和性能分析。请仔细分析所提供的EKS集群中同一Service下不同Pod的CPU使用率监控折线图，重点关注：
        1. 对比不同Pod之间的负载差异，如果有Pod与其他Pod有明显不同的CPU使用模式则视作异常Pod,注意缺乏正常负载波动特征的Pod如死平，长时间高负载等情况也视作异常Pod。
        2. 结合纵轴数值范围，折线图中明显偏离大部分pod的折线图对应的pod也视作异常Pod。
        3. 忽略CPU使用率瞬间异常峰值或谷值，这有对应的监控告警。
        4. 返回异常Pod名称，如果没有异常的pod则返回没有。不要解释。
        请返回有异常pod名称格式如<example>，无需返回原因，解决方案或建议等多余的内容。
        输出格式参考：
        <example1>
            <result>
            无异常
            </result>
        </example1>
        <example2>
            <result>
                <pod1>
                a-b-f44568697-g8rz8-172.24.154.112
                </pod1>
                <pod2>
                a-b-f44568697-g8rz8-172.24.154.112
                </pod2>
            </result>
        </example2>
        """
    
    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_template
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": encoded_image
                        }
                    }
                ]
            }
        ]
    }
    
    body = json.dumps(prompt_config)
    contentType = "application/json"
    accept = "application/json"
    
    response = bedrock_client.invoke_model(
        modelId=MODEL_ID,
        contentType=contentType,
        accept=accept,
        body=body
    )
    response_body = json.loads(response.get("body").read())
    return response_body.get("content")[0]["text"]

def lambda_handler(event, context):
    """
    Lambda function handler that processes CPU usage data from S3
    """
    try:
        # Get bucket and key from S3 event
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']
        
        print(f"Processing file {key} from bucket {bucket}")
        
        # Read data from S3
        df = read_cpu_data_from_s3(bucket, key)
        
        # Group data
        grouped_data = group_cpu_data(df)
        
        # Calculate statistics
        # stats = calculate_statistics(grouped_data)
        
        # Plot data and get service-grouped data
        service_groups = plot_cpu_usage(grouped_data)
        
        # Analyze data for each service
        all_normal = True
        
        for service, pod_data in service_groups.items():
            print(f"\nAnalyzing service: {service}")
            
            # Analyze the data
            data_analysis = llm_analysis(pod_data, service)
            data_result = extract_result_content(data_analysis)
            
            # Analyze the plot
            plot_analysis = llm_analysis_img(service)
            plot_result = extract_result_content(plot_analysis)
            
            # Check for anomalies
            anomaly_pods = []
            if "无异常" not in data_result:
                anomaly_pods.extend(extract_pod_names(data_analysis))
            if "无异常" not in plot_result:
                anomaly_pods.extend(extract_pod_names(plot_analysis))
            
            if anomaly_pods:
                all_normal = False
                # Upload image to S3 if anomalies are detected
                image_path = os.path.join(TMP_DIR, f"cpu_usage_{service}.jpg")
                with open(image_path, 'rb') as image_file:
                    s3_client.put_object(
                        Bucket=bucket,
                        Key=f"output/anomalies/cpu_usage_{service}.jpg",
                        Body=image_file,
                        ContentType='image/jpeg'
                    )
                print(f"Found anomalies in service {service}:")
                for pod in set(anomaly_pods):  # Remove duplicates
                    print(f"- {pod}")
                print(f"Uploaded anomaly plot for {service} to S3")
        
        if all_normal:
            print("\n所有服务均无异常")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Analysis completed successfully',
                'all_normal': all_normal
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
