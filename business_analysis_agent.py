import pandas as pd
import glob
from typing import List, Dict
from openai import OpenAI
import os

# 初始化 OpenAI 客户端
client = OpenAI()

# --------------------- DataAgent ---------------------
class DataAgent:
    """负责从多来源采集数据，包括 CSV、Excel、PDF 和文本会议纪要"""
    def __init__(self, data_paths: Dict[str, str]):
        self.data_paths = data_paths

    def load_csv_excel(self, path_pattern: str) -> List[pd.DataFrame]:
        dfs = []
        for file in glob.glob(path_pattern):
            if file.endswith('.csv'):
                dfs.append(pd.read_csv(file))
            elif file.endswith(('.xls', '.xlsx')):
                dfs.append(pd.read_excel(file))
        return dfs

    def load_text(self, path_pattern: str) -> List[str]:
        texts = []
        for file in glob.glob(path_pattern):
            with open(file, 'r', encoding='utf-8') as f:
                texts.append(f.read())
        return texts

    def load_data(self) -> Dict[str, any]:
        data = {
            'daily': self.load_csv_excel(self.data_paths.get('daily', '')),
            'weekly': self.load_csv_excel(self.data_paths.get('weekly', '')),
            'ledger': self.load_csv_excel(self.data_paths.get('ledger', '')),
            'meetings': self.load_text(self.data_paths.get('meetings', ''))
        }
        return data

# --------------------- AnalysisAgent ---------------------
class AnalysisAgent:
    """异常发现与归因分析"""
    def detect_anomalies(self, data: Dict[str, any]) -> List[Dict]:
        anomalies = []
        for df in data.get('daily', []):
            if 'cost' in df.columns:
                df['cost_change'] = df['cost'].pct_change()
                for idx, row in df.iterrows():
                    if pd.notnull(row['cost_change']) and row['cost_change'] > 0.1:
                        anomalies.append({'type': 'cost_spike', 'date': row['date'], 'value': row['cost']})
        # 可扩展: 转化率下滑、库存异常、项目延期等规则
        return anomalies

    def attribute_causes(self, anomalies: List[Dict], data: Dict[str, any]) -> List[Dict]:
        results = []
        for anom in anomalies:
            prompt = f"分析以下异常: {anom}. 根据历史数据和业务台账: {data.get('ledger', [])}，找出可能原因。"
            response = client.chat.completions.create(
                model='gpt-4',
                messages=[{"role": "user", "content": prompt}]
            )
            results.append({'anomaly': anom, 'cause_analysis': response.choices[0].message.content})
        return results

# --------------------- StrategyAgent ---------------------
class StrategyAgent:
    """生成可执行策略"""
    def generate_actions(self, analyzed_data: List[Dict]) -> List[Dict]:
        actions = []
        for item in analyzed_data:
            prompt = f"根据异常 {item['anomaly']} 及原因分析 {item['cause_analysis']}，生成可执行行动建议。请分步骤列出。"
            response = client.chat.completions.create(
                model='gpt-4',
                messages=[{"role": "user", "content": prompt}]
            )
            actions.append({'anomaly': item['anomaly'], 'actions': response.choices[0].message.content})
        return actions

# --------------------- FollowupAgent ---------------------
class FollowupAgent:
    """执行跟踪与复盘，支持多轮更新"""
    def follow_up(self, actions: List[Dict], previous_reports: List[Dict]=None) -> List[Dict]:
        followups = []
        for act in actions:
            prompt = f"跟踪以下行动: {act['actions']}。生成执行进度与复盘报告。"
            if previous_reports:
                prompt += f" 参考之前复盘: {previous_reports}"
            response = client.chat.completions.create(
                model='gpt-4',
                messages=[{"role": "user", "content": prompt}]
            )
            followups.append({'anomaly': act['anomaly'], 'followup': response.choices[0].message.content})
        return followups

# --------------------- Controller ---------------------
class BusinessAnalysisAgent:
    """协调多 Agent 完成闭环"""
    def __init__(self, data_agent: DataAgent, analysis_agent: AnalysisAgent,
                 strategy_agent: StrategyAgent, followup_agent: FollowupAgent):
        self.data_agent = data_agent
        self.analysis_agent = analysis_agent
        self.strategy_agent = strategy_agent
        self.followup_agent = followup_agent

    def run_closed_loop(self, previous_followups: List[Dict]=None):
        print("[Step 1] 数据采集")
        data = self.data_agent.load_data()

        print("[Step 2] 异常检测")
        anomalies = self.analysis_agent.detect_anomalies(data)

        print("[Step 3] 归因分析")
        analyzed = self.analysis_agent.attribute_causes(anomalies, data)

        print("[Step 4] 策略生成")
        actions = self.strategy_agent.generate_actions(analyzed)

        print("[Step 5] 执行跟进")
        followups = self.followup_agent.follow_up(actions, previous_reports=previous_followups)

        return followups

# --------------------- 使用示例 ---------------------
data_paths = {
    'daily': 'data/daily/*.csv',
    'weekly': 'data/weekly/*.xlsx',
    'ledger': 'data/ledger/*.csv',
    'meetings': 'data/meetings/*.txt'
}

data_agent = DataAgent(data_paths)
analysis_agent = AnalysisAgent()
strategy_agent = StrategyAgent()
followup_agent = FollowupAgent()

agent = BusinessAnalysisAgent(data_agent, analysis_agent, strategy_agent, followup_agent)
results = agent.run_closed_loop()

for r in results:
    print(r)
