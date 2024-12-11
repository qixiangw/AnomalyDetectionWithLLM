import streamlit as st
import boto3
import json
import pandas as pd
from botocore.exceptions import ClientError
from data_processor import analyze_time_series, get_column_mappings

# Initialize Streamlit app
st.title("Log Analyzer")

# Create sidebar for page selection
page = st.sidebar.selectbox("选择功能", ["CloudWatch日志分析", "Excel日志分析"])

# Initialize Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")

if page == "CloudWatch日志分析":
    # Create a session client for CloudWatch Logs
    session = boto3.Session()
    available_regions = []
    logs_client = session.client('logs')
    try:
        response = logs_client.describe_log_groups()
        available_regions = [region for region in session.get_available_regions('logs')]
    except ClientError as e:
        st.error(f"Error retrieving available regions: {e}")

    default_region = 'us-east-1' if 'us-east-1' in available_regions else available_regions[0] if available_regions else None

    if available_regions:
        selected_region = st.selectbox("Select a region", available_regions, index=available_regions.index(default_region) if default_region else 0)
        logs_client = session.client('logs', region_name=selected_region)

        try:
            log_groups = logs_client.describe_log_groups()['logGroups']
        except ClientError as e:
            st.error(f"Error retrieving log groups in {selected_region}: {e}")
            log_groups = []

        if not log_groups:
            st.warning(f"No logs available in {selected_region}, please select another region.")
        else:
            selected_log_group = st.selectbox("Select a log group", [log_group['logGroupName'] for log_group in log_groups])

            try:
                log_streams = logs_client.describe_log_streams(logGroupName=selected_log_group)['logStreams']
            except ClientError as e:
                st.error(f"Error retrieving log streams in {selected_region}/{selected_log_group}: {e}")
                log_streams = []

            if not log_streams:
                st.warning(f"No log streams available in {selected_region}/{selected_log_group}, please select another log group.")
            else:
                selected_log_stream = st.selectbox("Select a log stream", [log_stream['logStreamName'] for log_stream in log_streams])

                try:
                    log_stream_data = logs_client.get_log_events(
                        logGroupName=selected_log_group,
                        logStreamName=selected_log_stream
                    )

                    prompt_data = ''.join([event['message'] + '\n' for event in log_stream_data['events']])

                    st.markdown("**Log Data:**")
                    st.code(prompt_data, language="text")
                    
                    prompt_config = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1000,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"You are a threat detection expert working with the AWS Cloud. Review the log data I am providing for AWS service and explain in detail what you see, and if you have any recommendations on rules that could be configured based on what you see in the log data, please tell me in Chinese.\n\n {prompt_data}"
                                    }
                                ]
                            }
                        ]
                    }
                    
                    response = bedrock.invoke_model(
                        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                        contentType="application/json",
                        accept="application/json",
                        body=json.dumps(prompt_config)
                    )
                    
                    response_body = json.loads(response.get("body").read())
                    analysis = response_body.get("content")[0]["text"]
                    
                    if analysis:
                        st.markdown("**分析结果:**")
                        st.write(analysis)
                except ClientError as e:
                    st.error(f"Error retrieving log events in {selected_region}/{selected_log_group}/{selected_log_stream}: {e}")
    else:
        st.warning("No regions available with CloudWatch Logs access.")

elif page == "Excel日志分析":
    st.subheader("Excel日志分析")
    
    uploaded_file = st.file_uploader("上传Excel/CSV日志文件", type=['xlsx', 'xls', 'csv'])
    
    if uploaded_file is not None:
        try:
            # Read Excel or CSV file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Get column mappings and check if we have all required fields
            col_map = get_column_mappings(df)
            required_fields = ['service', 'pod', 'metric', 'node', 'timestamp', 'cpu_usage']
            missing_fields = [field for field in required_fields if field not in col_map]
            
            if missing_fields:
                st.error(f"无法识别以下必需字段: {', '.join(missing_fields)}")
                st.write("文件必须包含以下信息: 服务名称, Pod名称, 指标名称, 节点信息, 时间戳, CPU使用率")
                st.write("当前文件包含以下列:")
                st.write(", ".join(df.columns.tolist()))
            else:
                # Display the data preview
                st.write("文件内容预览:")
                st.dataframe(df.head())
                
                # Perform time series analysis
                st.subheader("时序数据分析")
                results = analyze_time_series(df, bedrock)
                
                # Convert results to JSON string for frontend
                results_json = json.dumps(results)
                
                # Use st.components.v1.html to inject custom JavaScript for visualization
                st.components.v1.html(f"""
                    <div id="visualization-container"></div>
                    <script>
                        // Store the analysis results in a global variable
                        window.analysisResults = {results_json};
                        
                        // You can access the data in frontend JavaScript as:
                        // window.analysisResults.visualization_data - for chart data
                        // window.analysisResults.analysis_results - for analysis results
                    </script>
                """, height=0)
                
                # Display analysis results in text form
                for result in results['analysis_results']:
                    st.subheader(f"分析结果 - {result['group']}")
                    st.write(result['analysis'])
                
        except Exception as e:
            st.error(f"处理文件时出错: {str(e)}")
