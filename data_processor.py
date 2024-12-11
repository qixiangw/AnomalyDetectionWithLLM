import pandas as pd
import os
import base64
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import random
from datetime import datetime, timedelta
import os
import boto3
import json

'''
未来可以做多个时段pod异常才认为是异常
img分析的pe可以加上死平，趋势等作为例子
'''

# Global constants
MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"

def read_cpu_data(file_path: str) -> pd.DataFrame:
    """
    Read CPU usage data from CSV file with error handling and validation
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        required_columns = ['service', 'pod', 'node', 'timestamp', 'cpuusage']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Validate CPU usage values
        '''
        if not all(0 <= x <= 100 for x in df['cpuusage']):
            print(df['cpuusage'])
            raise ValueError("CPU usage values must be between 0 and 100")
        '''
        return df
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty")
    except pd.errors.ParserError:
        raise ValueError("Error parsing CSV file. Please check the file format")

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
    Plot CPU usage for each service in a separate figure and save as jpg
    Using 2-hour window while preserving all original data points
    Returns: Dictionary with service as key and list of (pod, node, data) tuples as value
    """
    # Group by service first
    service_groups = {}
    for (service, pod, node), data in grouped_data.items():
        if service not in service_groups:
            service_groups[service] = []
        service_groups[service].append((pod, node, data))
    
    # Create a separate figure for each service
    for service, pod_data in service_groups.items():
        plt.figure(figsize=(15, 8))
        
        # Get the earliest and latest timestamps across all pods for this service
        all_timestamps = []
        for pod, node, data in pod_data:
            all_timestamps.extend(data['timestamp'])
        min_timestamp = min(all_timestamps)
        
        # Select a 2-hour window
        window_start = min_timestamp
        window_end = window_start + timedelta(hours=2)
        
        # Store window data for each pod
        window_pod_data = []
        for pod, node, data in pod_data:
            # Filter data for the 2-hour window
            mask = (data['timestamp'] >= window_start) & (data['timestamp'] <= window_end)
            window_data = data[mask]
            window_pod_data.append((pod, node, window_data))
            
            # Plot all original data points within the window
            label = f"{pod}-{node}"
            plt.plot(window_data['timestamp'], window_data['cpuusage'], label=label, marker='o', markersize=4)
        
        # Update service_groups with window data
        service_groups[service] = window_pod_data
        
        plt.xlabel('Time')
        plt.ylabel('CPU Usage (%)')
        plt.title(f'CPU Usage Over Time - {service}\n(2-hour window with all data points)')
        plt.grid(True)
        
        # Set x-axis ticks to show reasonable time intervals
        time_interval = timedelta(minutes=15)  # Show ticks every 15 minutes
        ticks = pd.date_range(start=window_start, end=window_end, freq=time_interval)
        plt.xticks(ticks, [t.strftime('%Y-%m-%d %H:%M') for t in ticks], rotation=45)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the figure as jpg
        filename = f"cpu_usage_{service}.jpg"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()  # Close the figure to free memory
        print(f"Saved plot for {service} as {filename}")
    
    return service_groups

def llm_analysis(service_data: List[Tuple[str, str, pd.DataFrame]], service_name: str):
    """
    Analyze CPU usage data for a specific service
    Args:
        service_data: List of (pod, node, data) tuples for the service
        service_name: Name of the service being analyzed
    """
    bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
    
    # Format the data for analysis
    analysis_text = f"Service: {service_name}\n\n"
    
    # Calculate statistics for each pod
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
    
    # Add raw data
    raw_data_text = "Raw Data:\n"
    for pod, node, data in service_data:
        raw_data_text += f"Pod: {pod}\nNode: {node}\n"
        raw_data_text += f"Data:\n{data['cpuusage'].to_string()}\n\n"
    
    prompt_template = """你是一位拥有丰富AWS云平台运维经验的DevOps专家，特别擅长容器以及微服务中的日志和性能分析。请仔细分析所提供的EKS集群中同一Service下不同Pod的{metic}CPU使用率日志<log>，重点关注：
        1. 对比不同Pod之间的负载差异，如果有Pod与其他Pod有明显不同的CPU使用模式则视作异常Pod,注意缺乏正常负载波动特征的Pod如死平，长时间高负载等情况也视作异常Pod；
        2. 参考<data>中监控指标的统计数值，有均值，最值和方差，分析这些统计指标来辅助识别异常的Pod，比如某个pod的平均值偏离多数pod的平均值就是异常Pod；
        3. 忽略CPU使用率瞬间异常峰值或谷值，这有对应的监控告警。
        4. 返回异常Pod名称，如果没有异常的pod则返回没有。不要解释。
        <rule>
        1. 
        2. 
        </rule>
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
    
    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType=contentType,
        accept=accept,
        body=body
    )
    response_body = json.loads(response.get("body").read())
    summary = response_body.get("content")[0]["text"]
    return summary

def llm_analysis_img(image_path: str) -> str:
    """
    Analyze CPU usage anomalies from a monitoring plot image using Claude
    Args:
        image_path: Path to the CPU usage plot image
    Returns:
        str: Analysis summary in Chinese
    """
    bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
    
    # Read and encode the image
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
    
    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType=contentType,
        accept=accept,
        body=body
    )
    response_body = json.loads(response.get("body").read())
    summary = response_body.get("content")[0]["text"]
    return summary

def main(csv_file_path: str):
    try:
        # Read data
        print(f"Reading data from {csv_file_path}")
        df = read_cpu_data(csv_file_path)
        
        # Group data
        print("\nGrouping data by service, pod, and node...")
        grouped_data = group_cpu_data(df)
        
        # Calculate statistics
        print("\nCalculating statistics for each group:")
        stats = calculate_statistics(grouped_data)
        for key, stat in stats.items():
            print(f"\n{key}:")
            for metric, value in stat.items():
                print(f"  {metric}: {value:.2f}")
        
        # Plot data and get service-grouped data
        print("\nGenerating plots...")
        service_groups = plot_cpu_usage(grouped_data)
        
        # Analyze data for each service using both methods
        print("\nAnalyzing services...")
        for service, pod_data in service_groups.items():
            print(f"\nAnalyzing service: {service}")
            
            # Analyze the resampled data
            # print("\nAnalyzing CPU usage data:")
            analysis = llm_analysis(pod_data, service)
            print("\nData Analysis result:")
            print(analysis)
            
            # Analyze the plot
            image_path = f"cpu_usage_{service}.jpg"
            print(f"\nAnalyzing plot {image_path}...")
            plot_analysis = llm_analysis_img(image_path)
            print("\nPlot Analysis result:")
            print(plot_analysis)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    csv_file_path = "tsdb-k.csv"
    main(csv_file_path)
