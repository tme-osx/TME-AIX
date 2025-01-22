# Author: Fatih E. NAR
# 2025 Texas US
# Enhanced with GPU acceleration and improved visualizations

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import plotly.express as px
from sklearn.ensemble import IsolationForest
import warnings
import os
import pickle
import torch
from accelerate import Accelerator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

warnings.filterwarnings('ignore')

print("Starting Enhanced 5G Core Network Operations Center...")

# Initialize accelerator for GPU support
accelerator = Accelerator()
print(f"Using device: {accelerator.device}")

# Custom JSON encoder for timestamp handling
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super().default(obj)

class NOCState:
    def __init__(self):
        self.cumulative_anomalies = {'AMF': 0, 'SMF': 0, 'UPF': 0}
        self.anomalies = {'AMF': 0, 'SMF': 0, 'UPF': 0}
        self.alert_history = []
        self.recommendations_history = []
        self.resolution_history = []
        self.processed_anomalies = set()
        self.alert_scroll_position = 0
        self.recommendation_scroll_position = 0
        self.rca_history = []

    def update_rca(self, rca_result, current_datetime):
        if isinstance(current_datetime, str):
            timestamp = pd.to_datetime(current_datetime)
        else:
            timestamp = current_datetime

        # Only add RCA entries that indicate actual issues
        if "No active issues detected" not in rca_result['rca'] and \
        "Network operating within normal parameters" not in rca_result['recommendations']:
            # Ensure timestamp is always a pandas Timestamp
            rca_entry = {
                'timestamp': pd.to_datetime(timestamp) if isinstance(timestamp, str) else timestamp,
                'rca': rca_result['rca'],
                'recommendations': rca_result['recommendations']
            }

            # Check if we already have this entry
            exists = any(
                entry['timestamp'] == timestamp and
                entry['rca'] == rca_result['rca']
                for entry in self.rca_history
            )

            if not exists:
                self.rca_history.append(rca_entry)
                # Sort by timestamp, most recent first
                self.rca_history.sort(key=lambda x: x['timestamp'], reverse=True)

    def update_anomaly_counts(self, component, timestamp, anomaly_id):
        anomaly_key = f"{component}_{timestamp}_{anomaly_id}"
        if anomaly_key not in self.processed_anomalies:
            self.cumulative_anomalies[component] += 1
            self.processed_anomalies.add(anomaly_key)
    
    def _calculate_severity_level(self, alert):
        severity_mapping = {
            'CRITICAL': 3,
            'MAJOR': 2,
            'MINOR': 1
        }
        return severity_mapping.get(alert.get('severity', 'MINOR'), 1)

    def update_alerts(self, alert, current_datetime):
        alert_key = f"{alert['component']}_{alert['type']}_{alert['start_time']}_{alert['end_time']}"
        if alert_key not in [a['key'] for a in self.alert_history]:
            if isinstance(current_datetime, (datetime, pd.Timestamp)):
                first_seen = current_datetime.isoformat()
            else:
                first_seen = pd.to_datetime(current_datetime).isoformat()
                
            self.alert_history.append({
                'key': alert_key,
                'alert': alert,
                'first_seen': first_seen,
                'resolved': False,
                'resolution_time': None,
                'resolution_details': None,
                'severity_level': self._calculate_severity_level(alert)
            })

    def resolve_alert(self, alert_key, current_datetime, resolution_details):
        for alert in self.alert_history:
            if alert['key'] == alert_key and not alert['resolved']:
                if isinstance(current_datetime, str):
                    resolution_time = pd.to_datetime(current_datetime)
                else:
                    resolution_time = pd.to_datetime(current_datetime)
                
                alert['resolved'] = True
                alert['resolution_time'] = resolution_time.isoformat()
                alert['resolution_details'] = resolution_details
                
                alert['_resolution_time_obj'] = resolution_time
                alert['_first_seen_obj'] = pd.to_datetime(alert['first_seen'])
                
                self.resolution_history.append({
                    'alert': alert['alert'],
                    'resolution_time': resolution_time.isoformat(),
                    'resolution_details': resolution_details,
                    'resolution_metrics': self._capture_resolution_metrics(alert)
                })

    def _capture_resolution_metrics(self, alert):
        try:
            first_seen = pd.to_datetime(alert['first_seen'])
            resolution_time = pd.to_datetime(alert['resolution_time'])
            time_to_resolve = (resolution_time - first_seen).total_seconds()
        except Exception as e:
            print(f"Error calculating resolution time: {str(e)}")
            time_to_resolve = 0
            
        return {
            'time_to_resolve': time_to_resolve,
            'severity': alert.get('severity_level', 1),
            'component': alert['alert']['component']
        }

    @classmethod
    def from_dict(cls, state_dict):
        state = cls()
        if state_dict:
            state.cumulative_anomalies = state_dict.get('cumulative_anomalies', {'AMF': 0, 'SMF': 0, 'UPF': 0})
            state.processed_anomalies = set(state_dict.get('processed_anomalies', []))
            state.alert_history = [
                {
                    'key': alert['key'],
                    'alert': alert['alert'],
                    'first_seen': pd.to_datetime(alert['first_seen']) if alert.get('first_seen') else None,
                    'resolved': alert['resolved'],
                    'resolution_time': pd.to_datetime(alert['resolution_time']) if alert.get('resolution_time') else None,
                    'resolution_details': alert['resolution_details'],
                    'severity_level': alert.get('severity_level', 1)
                }
                for alert in state_dict.get('alert_history', [])
            ]
            state.resolution_history = state_dict.get('resolution_history', [])
            state.rca_history = state_dict.get('rca_history', [])
        return state

# Helper Functions
def load_data():
    print(f"Loading data from {os.getcwd()}/data2 directory...")
    
    @accelerator.on_main_process
    def load_csv(file_path):
        return pd.read_csv(file_path)
    
    amf_df = load_csv('data2/amf_metrics.csv')
    smf_df = load_csv('data2/smf_metrics.csv')
    upf_df = load_csv('data2/upf_metrics.csv')
    
    for df in [amf_df, smf_df, upf_df]:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
    
    return amf_df, smf_df, upf_df

def load_alerts():
    print("Loading alerts data...")
    with open('data2/alerts.json', 'r') as f:
        return json.load(f)

@accelerator.on_main_process
def load_preprocessed_data():
    print("Loading preprocessed data...")
    
    embeddings = OpenAIEmbeddings()
    vector_stores = {}
    
    try:
        for component in ['amf', 'smf', 'upf']:
            vector_stores[component] = FAISS.load_local(
                f"processed_data2/{component}_vector_store",
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Successfully loaded {component} vector store")
            
            with open(f"processed_data2/{component}_documents.pkl", 'rb') as f:
                docs = pickle.load(f)
                print(f"Loaded {len(docs)} documents for {component}")
    except Exception as e:
        print(f"Warning: Could not load vector stores: {str(e)}")
        vector_stores = {}
    
    try:
        with open('processed_data2/anomalies.pkl', 'rb') as f:
            anomalies = pickle.load(f)
        print("Successfully loaded anomalies")
    except Exception as e:
        print(f"Warning: Could not load anomalies: {str(e)}")
        anomalies = {'amf': [], 'smf': [], 'upf': []}
    
    try:
        with open('processed_data2/rca_results.pkl', 'rb') as f:
            rca_results = pickle.load(f)
        
        # Process RCA results for better timestamp handling
        rca_by_timestamp = {}
        for result in rca_results:
            try:
                timestamp = pd.to_datetime(result['timestamp'])
                rca_by_timestamp[timestamp] = {
                    'rca': result.get('rca', '...'),
                    'recommendations': result.get('recommendations', '...'),
                    'confidence': result.get('confidence', 'medium'),
                    'component_impacts': result.get('component_impacts', {}),
                    'metrics_impact': result.get('metrics_impact', {})
                }
            except Exception as e:
                print(f"Error processing RCA result: {str(e)}")
                continue
                
        print(f"Successfully loaded {len(rca_results)} RCA results")
    except Exception as e:
        print(f"Warning: Could not load RCA results: {str(e)}")
        rca_by_timestamp = {}
    
    return vector_stores, anomalies, rca_by_timestamp

def create_alerts_display(noc_state, current_datetime):
    if not noc_state.alert_history:
        return html.Div([
            html.P("No active alerts", className="text-muted text-center my-3")
        ], className="alert-container")

    def get_severity_badge(severity):
        colors = {
            'CRITICAL': 'danger',
            'MAJOR': 'warning',
            'MINOR': 'info'
        }
        return colors.get(severity, 'secondary')

    alert_items = []
    sorted_alerts = sorted(
        noc_state.alert_history,
        key=lambda x: (
            not x['resolved'],  # Active alerts first
            -x.get('severity_level', 1),  # Higher severity first
            pd.Timestamp(x['first_seen'])  # Most recent first
        )
    )

    for alert in sorted_alerts:
        severity = alert['alert'].get('severity', 'MINOR')
        status_class = "text-danger fw-bold" if not alert['resolved'] else "text-success"
        
        alert_items.append(
            html.Div([
                html.Div([
                    html.Span(
                        f"{alert['alert']['component']} - {alert['alert']['type']}",
                        className=f"badge bg-{get_severity_badge(severity)} me-2"
                    ),
                    html.Small(
                        f"Started: {pd.Timestamp(alert['first_seen']).strftime('%H:%M:%S')}",
                        className="text-muted me-2"
                    ),
                    html.Span(
                        f"Status: {'ACTIVE' if not alert['resolved'] else 'RESOLVED'}",
                        className=status_class
                    )
                ], className="d-flex align-items-center"),
                html.Div([
                    html.P(alert['alert']['description'], className="mb-1 mt-2"),
                    html.Small(
                        [
                            "Resolution: ",
                            html.Span(
                                alert.get('resolution_details', 'Pending...'),
                                className="text-success" if alert['resolved'] else "text-muted"
                            )
                        ],
                        className="d-block"
                    ) if alert['resolved'] else None
                ], className="alert-details")
            ], className="alert border-start border-4 border-secondary p-3 mb-2")
        )

    return html.Div([
        html.Div(
            alert_items,
            className="alerts-scroll",
            style={
                'maxHeight': '250px',
                'overflowY': 'auto',
                'scrollBehavior': 'smooth'
            }
        ),
        html.Div([
            html.Small([
                f"Total: {len(noc_state.alert_history)} | ",
                html.Span(
                    f"Active: {len([a for a in noc_state.alert_history if not a['resolved']])}",
                    className="text-danger"
                ),
                " | ",
                html.Span(
                    f"Resolved: {len([a for a in noc_state.alert_history if a['resolved']])}",
                    className="text-success"
                )
            ], className="text-muted")
        ], className="mt-2 text-center")
    ])

def create_combined_rca_display(rca_history):
    def highlight_keywords(text):
        keywords = {
            "Root Cause Analysis": "text-primary fw-bold",
            "Recommendations": "text-success fw-bold",
            "Immediate actions": "text-danger fw-bold",
            "Prevention measures": "text-warning fw-bold",
            "Monitoring suggestions": "text-info fw-bold"
        }
        
        elements = []
        current_text = text.strip()
        
        for keyword, style_class in keywords.items():
            if keyword in current_text:
                parts = current_text.split(keyword)
                for i, part in enumerate(parts):
                    if i > 0:  # Add keyword with highlighting
                        elements.append(html.Span(f"\n{keyword}:\n", className=style_class))
                    if part:  # Add regular text
                        cleaned_part = part.replace(":\n\n", "").strip()
                        if cleaned_part:
                            elements.append(html.Span(cleaned_part))
                current_text = "".join(parts)
        
        if not elements:  # If no keywords found, return original text
            elements = [html.Span(current_text)]
            
        return html.Pre(
            elements,
            style={
                'white-space': 'pre-wrap',
                'margin': '0',
                'padding': '10px',
                'backgroundColor': 'rgba(240,240,240,0.5)',
                'borderRadius': '5px',
                'fontSize': '0.9rem',
                'lineHeight': '1.5'
            }
        )

    return html.Div([
        html.H4("Root Cause Analysis & Recommendations with GenAI", 
                className="card-header bg-primary text-white"),
        html.Div([
            html.Div(
                [
                    html.Div([
                        html.Small(
                            pd.to_datetime(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                            className="text-muted d-block mb-2"
                        ),
                        highlight_keywords(
                            f"Root Cause Analysis\n{entry['rca'].strip()}\n\n"
                            f"Recommendations\n{entry['recommendations'].strip()}"
                        ),
                        html.Hr(className="my-3")
                    ], key=f"{entry['timestamp']}_{hash(entry['rca'])}")
                    for entry in rca_history
                ] if rca_history else [
                    html.P("Awaiting Analysis...", 
                          className="text-muted text-center")
                ],
                style={
                    'maxHeight': 'calc(800px - 56px)',
                    'overflowY': 'auto',
                    'scrollBehavior': 'smooth',
                    'padding': '15px'
                },
                id='rca-history-container',
                className="rca-history"
            )
        ], className="card-body p-0")
    ], className="card shadow-sm h-100")

def get_current_rca(rca_by_timestamp, current_datetime, window_minutes=5):
    if not rca_by_timestamp:
        return {
            'rca': "No historical RCA data available",
            'recommendations': "System is collecting data for analysis"
        }
    
    # Ensure current_datetime is a pandas Timestamp
    if not isinstance(current_datetime, pd.Timestamp):
        current_datetime = pd.to_datetime(current_datetime)
    
    window_start = current_datetime - timedelta(minutes=window_minutes)
    window_end = current_datetime + timedelta(minutes=window_minutes)
    
    closest_rca = None
    min_diff = timedelta(minutes=window_minutes)
    
    # Convert all timestamps to pandas Timestamp for comparison
    for timestamp, rca_data in rca_by_timestamp.items():
        # Ensure timestamp is a pandas Timestamp
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.to_datetime(timestamp)
            
        if window_start <= timestamp <= window_end:
            time_diff = abs(timestamp - current_datetime)
            if time_diff < min_diff:
                closest_rca = rca_data
                min_diff = time_diff
    
    if closest_rca:
        return closest_rca
    
    return {
        'rca': "No active issues detected",
        'recommendations': "Network operating within normal parameters"
    }

def create_metric_figure(component, df, current_datetime, anomalies_info=None):
    window_start = current_datetime - timedelta(minutes=30)
    df_window = df[(df['timestamp'] >= window_start) & (df['timestamp'] <= current_datetime)]
    
    fig = go.Figure()
    
    metrics = {
        'AMF': [
            ('registration_rate', 'Registration Rate', '#1f77b4'),
            ('registration_success_rate', 'Registration Success Rate', '#2ca02c'),
            ('authentication_success_rate', 'Auth Success Rate', '#ff7f0e')
        ],
        'SMF': [
            ('session_establishment_rate', 'Session Est. Rate', '#1f77b4'),
            ('session_success_rate', 'Session Success Rate', '#2ca02c'),
            ('ip_pool_utilization', 'IP Pool Usage', '#ff7f0e')
        ],
        'UPF': [
            ('active_bearers', 'Active Bearers', '#1f77b4'),
            ('throughput_mbps', 'Throughput (Mbps)', '#2ca02c'),
            ('latency_ms', 'Latency (ms)', '#ff7f0e')
        ]
    }[component]

    for metric, name, color in metrics:
        try:
            # Move data to GPU if available
            if torch.cuda.is_available():
                metric_data = torch.tensor(df_window[metric].values, device=accelerator.device)
                timestamps = df_window['timestamp'].values
            else:
                metric_data = df_window[metric].values
                timestamps = df_window['timestamp'].values

            fig.add_trace(go.Scatter(
                x=timestamps,
                y=metric_data.cpu().numpy() if torch.is_tensor(metric_data) else metric_data,
                name=name,
                line=dict(color=color),
                mode='lines+markers',
                marker=dict(size=6, symbol='circle')
            ))
            
            if anomalies_info:
                anomaly_data = [(entry['timestamp'], entry['anomalies'].get(metric, {}))
                               for entry in anomalies_info
                               if metric in entry['anomalies'] and 
                               entry['timestamp'] in df_window['timestamp'].values]
                
                if anomaly_data:
                    valid_data = [(t, v) for t, v in anomaly_data if v]
                    if valid_data:
                        times, values = zip(*[(t, v['value']) for t, v in valid_data])
                        hover_texts = [
                            f"{name}<br>"
                            f"Value: {v['value']:.2f}<br>"
                            f"Mean: {v['mean']:.2f}<br>"
                            f"Std Dev: {v['std']:.2f}<br>"
                            f"Deviation: {v.get('deviation_sigma', 0):.2f}σ"
                            for _, v in valid_data
                        ]
                        
                        fig.add_trace(go.Scatter(
                            x=times,
                            y=values,
                            mode='markers',
                            marker=dict(
                                size=12,
                                symbol='star',
                                color='red',
                                line=dict(width=2, color='red')
                            ),
                            name=f'{name} Anomalies',
                            text=hover_texts,
                            hoverinfo='text+x',
                            showlegend=False
                        ))
        except Exception as e:
            print(f"Error plotting metric {metric} for {component}: {str(e)}")
            continue

    fig.update_layout(
        title=dict(
            text=f'{component} Metrics',
            x=0.01,
            y=0.95,
            font=dict(size=16, color='#2c3e50')
        ),
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        xaxis_range=[window_start, current_datetime],
        hovermode='x unified',
        plot_bgcolor='rgba(240,240,240,0.5)',
        paper_bgcolor='rgba(255,255,255,0.5)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(128,128,128,0.2)',
            title_font=dict(size=12)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(128,128,128,0.2)',
            title_font=dict(size=12)
        ),
        modebar=dict(
            bgcolor='rgba(255,255,255,0.7)',
            color='#2c3e50'
        )
    )
    
    return fig.to_dict()

def create_anomaly_figure(anomaly_data):
    try:
        if not anomaly_data:
            return go.Figure().to_dict()

        # Ensure all components exist with at least 0 count
        complete_data = {
            'AMF': 0,
            'SMF': 0,
            'UPF': 0,
            **{k: v for k, v in anomaly_data.items() if k in ['AMF', 'SMF', 'UPF']}
        }

        df = pd.DataFrame.from_dict(
            complete_data, 
            orient='index',
            columns=['Count']
        ).reset_index()
        df.columns = ['Component', 'Count']
        
        colors = {
            'AMF': '#1f77b4',
            'SMF': '#ff7f0e',
            'UPF': '#9467bd'
        }
        
        fig = go.Figure()
        
        for component in ['AMF', 'SMF', 'UPF']:
            count = df[df['Component'] == component]['Count'].values[0]
            fig.add_trace(go.Bar(
                x=[component],
                y=[count],
                name=component,
                marker_color=colors.get(component, '#333333'),
                text=[f"{count}"],
                textposition='auto',
                hovertemplate="<b>%{x}</b><br>Anomalies: %{y}<extra></extra>"
            ))
        
        # Get the maximum count for y-axis range
        max_count = df['Count'].max()
        y_max = max(max_count * 1.2, 1)  # At least 1 or 20% higher than max
        
        fig.update_layout(
            title=dict(
                text='Network Component Anomalies',
                x=0.5,
                y=0.95,
                font=dict(size=16, color='#2c3e50')
            ),
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False,
            plot_bgcolor='rgba(240,240,240,0.5)',
            paper_bgcolor='rgba(255,255,255,0.5)',
            xaxis=dict(
                title="",
                showgrid=False,
                showline=True,
                linewidth=1,
                linecolor='rgba(128,128,128,0.2)'
            ),
            yaxis=dict(
                title="Total Anomalies",
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                showline=True,
                linewidth=1,
                linecolor='rgba(128,128,128,0.2)',
                zeroline=True,
                zerolinecolor='rgba(128,128,128,0.2)',
                range=[0, y_max],  # Force y-axis to start at 0
                dtick=1  # Set tick interval to 1
            ),
            modebar=dict(
                bgcolor='rgba(255,255,255,0.7)',
                color='#2c3e50'
            )
        )
        
        return fig.to_dict()
    except Exception as e:
        print(f"Error creating anomaly figure: {str(e)}")
        return go.Figure().to_dict()

# Initialize data
try:
    print("\nInitializing data...")
    amf_df, smf_df, upf_df = load_data()
    alerts = load_alerts()
    vector_stores, anomalies, rca_by_timestamp = load_preprocessed_data()
except Exception as e:
    print(f"Error during initialization: {str(e)}")
    raise

# Get time range for slider
min_time = min(amf_df['timestamp'].min(), smf_df['timestamp'].min(), upf_df['timestamp'].min())
max_time = max(amf_df['timestamp'].max(), smf_df['timestamp'].max(), upf_df['timestamp'].max())
print(f"\nTime range: {min_time} to {max_time}")

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        'https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css'
    ],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

# Enhanced layout with modern styling
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("5G Core Network Operations Center", className="text-center text-light p-3 mb-0")
    ], className="bg-dark shadow-sm"),
    
    # Playback controls
    html.Div([
        html.Div([
            html.Button('⏵️ Play', id='play-button', className="btn btn-primary me-2"),
            dcc.Dropdown(
                id='speed-control',
                options=[
                    {'label': '1x', 'value': 1},
                    {'label': '2x', 'value': 10},
                    {'label': '5x', 'value': 100},
                    {'label': '10x', 'value': 1000}
                ],
                value=1,
                style={'width': '100px'},
                className="d-inline-block me-2"
            ),
            html.Div(id='current-time', className="d-inline-block")
        ], className="d-flex align-items-center mb-2"),
        
        dcc.Slider(
            id='time-slider',
            min=0,
            max=(max_time - min_time).total_seconds(),
            value=0,
            marks={
                0: min_time.strftime('%Y-%m-%d %H:%M'),
                (max_time - min_time).total_seconds(): max_time.strftime('%Y-%m-%d %H:%M')
            },
            updatemode='drag',
            className="mb-3"
        )
    ], className="container-fluid bg-light p-3 shadow-sm"),
    
    # Main content
    html.Div([
        # Metrics row
        html.Div([
            html.Div([
                dcc.Graph(id='amf-metrics-graph', className="shadow-sm rounded")
            ], className="col-md-4"),
            html.Div([
                dcc.Graph(id='smf-metrics-graph', className="shadow-sm rounded")
            ], className="col-md-4"),
            html.Div([
                dcc.Graph(id='upf-metrics-graph', className="shadow-sm rounded")
            ], className="col-md-4")
        ], className="row mb-4"),

        # Analysis and Alerts row with RCA
        html.Div([
            # Left column with Anomaly Analysis and Active Alerts
            html.Div([
                # Anomaly Analysis Box
                html.Div([
                    html.Div([
                        html.H4("Anomaly Detection with Classic AI (NN)", className="card-header bg-primary text-white"),
                        html.Div([
                            dcc.Graph(id='anomaly-counts-graph')
                        ], className="card-body")
                    ], className="card shadow-sm h-100 mb-4")
                ]),
                # Active Alerts Box
                html.Div([
                    html.Div([
                        html.H4("Active Real Alerts from Systems", className="card-header bg-danger text-white"),
                        html.Div(
                            id='alerts-table',
                            className="card-body"
                        )
                    ], className="card shadow-sm h-100")
                ])
            ], className="col-md-6"),
            
            # Right column with RCA & Recommendations
            html.Div([
                html.Div(
                    id='rca-output',
                    className="h-100"  # Make it full height
                )
            ], className="col-md-6")
        ], className="row", style={'minHeight': '800px'}),  # Set minimum height for the row
        
        # Hidden divs for storing state
        html.Div(id='current-time-store', style={'display': 'none'}, children='0'),
        html.Div(id='play-store', style={'display': 'none'}, children='stopped'),
        html.Div(id='historical-state', style={'display': 'none'}, children='{}'),
        
        # Update interval
        dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
    ], className="container-fluid p-4")
])

# Callbacks
@app.callback(
    Output('play-store', 'children'),
    Input('play-button', 'n_clicks'),
    State('play-store', 'children')
)
def toggle_play(n_clicks, play_state):
    if n_clicks is None:
        return 'stopped'
    return 'playing' if play_state == 'stopped' else 'stopped'

@app.callback(
    Output('play-button', 'children'),
    Input('play-store', 'children')
)
def update_button_text(play_state):
    return '⏸️ Pause' if play_state == 'playing' else '⏵️ Play'

@app.callback(
    [Output('current-time-store', 'children'),
     Output('time-slider', 'value'),
     Output('current-time', 'children'),
     Output('amf-metrics-graph', 'figure'),
     Output('smf-metrics-graph', 'figure'),
     Output('upf-metrics-graph', 'figure'),
     Output('anomaly-counts-graph', 'figure'),
     Output('alerts-table', 'children'),
     Output('rca-output', 'children'),
     Output('historical-state', 'children')],  # removed recommendations.children
    [Input('interval-component', 'n_intervals'),
     Input('time-slider', 'value')],
    [State('play-store', 'children'),
     State('speed-control', 'value'),
     State('current-time-store', 'children'),
     State('historical-state', 'children')]
)
def update_dashboard(n_intervals, slider_value, play_state, speed, current_time, historical_state):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # For initial load, show empty state
    if not ctx.triggered:
        empty_state = {
            'anomalies': {'AMF': 0, 'SMF': 0, 'UPF': 0},
            'alert_history': [],
            'resolution_history': [],
            'rca_history': []
        }
        
        return (
            '0',  # current-time-store
            0,    # time-slider
            html.H4(min_time.strftime('%Y-%m-%d %H:%M:%S'), 
                   className="mb-0 ms-2 text-primary"),
            create_metric_figure('AMF', amf_df, min_time, []),
            create_metric_figure('SMF', smf_df, min_time, []),
            create_metric_figure('UPF', upf_df, min_time, []),
            create_anomaly_figure({'AMF': 0, 'SMF': 0, 'UPF': 0}),
            html.P("Monitoring system initialized...", 
                  className="text-muted text-center"),
            create_combined_rca_display([]),
            json.dumps(empty_state)
        )

    # Update current time based on trigger
    current_time = float(current_time)
    was_reset = False
    if trigger_id == 'interval-component' and play_state == 'playing':
        current_time = current_time + speed
        if current_time >= (max_time - min_time).total_seconds():
            current_time = 0
            was_reset = True  # Flag to indicate timeline reset
    elif trigger_id == 'time-slider':
        current_time = slider_value

    # Calculate current datetime
    current_datetime = min_time + timedelta(seconds=current_time)

    # Initialize or load NOC state
    try:
        if was_reset:
            # Preserve existing state on reset
            noc_state = NOCState.from_dict(json.loads(historical_state))
            # Clear only the temporary flags but keep historical data
            noc_state.anomalies = {'AMF': 0, 'SMF': 0, 'UPF': 0}
        else:
            noc_state = NOCState.from_dict(json.loads(historical_state)) if historical_state != '{}' else NOCState()
    except Exception as e:
        print(f"Error loading NOC state: {str(e)}")
        noc_state = NOCState()

    # Process historical data up to current time using GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        try:
            with torch.cuda.device(device):
                # Process anomalies while preserving history
                amf_anomalies = [entry for entry in anomalies['amf'] 
                                if entry['anomalies'] and pd.Timestamp(entry['timestamp']) <= current_datetime]
                smf_anomalies = [entry for entry in anomalies['smf'] 
                                if entry['anomalies'] and pd.Timestamp(entry['timestamp']) <= current_datetime]
                upf_anomalies = [entry for entry in anomalies['upf'] 
                                if entry['anomalies'] and pd.Timestamp(entry['timestamp']) <= current_datetime]

                # Update current anomaly counts
                current_anomalies = {
                    'AMF': len(amf_anomalies),
                    'SMF': len(smf_anomalies),
                    'UPF': len(upf_anomalies)
                }

                # On reset, preserve cumulative counts
                if not was_reset:
                    noc_state.anomalies = current_anomalies
                    for component, anomalies_list in [('AMF', amf_anomalies), ('SMF', smf_anomalies), ('UPF', upf_anomalies)]:
                        for anomaly in anomalies_list:
                            timestamp = pd.Timestamp(anomaly['timestamp'])
                            anomaly_id = hash(str(anomaly['anomalies']))
                            noc_state.update_anomaly_counts(component, timestamp, anomaly_id)
                            
                # Ensure GPU memory is cleared
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing anomalies on GPU: {str(e)}")
            torch.cuda.empty_cache()
            pass
    else:
        # CPU processing with similar preservation logic
        amf_anomalies = [entry for entry in anomalies['amf'] 
                         if entry['anomalies'] and pd.Timestamp(entry['timestamp']) <= current_datetime]
        smf_anomalies = [entry for entry in anomalies['smf'] 
                         if entry['anomalies'] and pd.Timestamp(entry['timestamp']) <= current_datetime]
        upf_anomalies = [entry for entry in anomalies['upf'] 
                         if entry['anomalies'] and pd.Timestamp(entry['timestamp']) <= current_datetime]

        current_anomalies = {
            'AMF': len(amf_anomalies),
            'SMF': len(smf_anomalies),
            'UPF': len(upf_anomalies)
        }

        if not was_reset:
            noc_state.anomalies = current_anomalies
            
    # Create visualization components
    window_size = timedelta(minutes=30)
    window_start = current_datetime - window_size
    
    window_anomalies = {
        'amf': [entry for entry in amf_anomalies 
                if pd.Timestamp(entry['timestamp']) >= window_start],
        'smf': [entry for entry in smf_anomalies 
                if pd.Timestamp(entry['timestamp']) >= window_start],
        'upf': [entry for entry in upf_anomalies 
                if pd.Timestamp(entry['timestamp']) >= window_start]
    }

    # Process alerts
    active_alerts = [
        alert for alert in alerts['alerts']
        if datetime.strptime(alert['start_time'], '%Y-%m-%d %H:%M:%S.%f') <= current_datetime
        and datetime.strptime(alert['end_time'], '%Y-%m-%d %H:%M:%S.%f') >= current_datetime
    ]

    for alert in active_alerts:
        noc_state.update_alerts(alert, current_datetime)

    # Process resolved alerts
    for alert in noc_state.alert_history:
        if not alert['resolved']:
            alert_end = datetime.strptime(alert['alert']['end_time'], '%Y-%m-%d %H:%M:%S.%f')
            if current_datetime > alert_end:
                noc_state.resolve_alert(
                    alert['key'],
                    current_datetime,
                    "Alert cleared - conditions normalized"
                )

    # Get RCA results
    try:
        # Create RCA history list
        current_rca_entries = []
        if not isinstance(current_datetime, pd.Timestamp):
            current_datetime = pd.to_datetime(current_datetime)
            
        for timestamp, rca_data in sorted(rca_by_timestamp.items(), reverse=True):
            # Ensure timestamp is a pandas Timestamp for comparison
            if not isinstance(timestamp, pd.Timestamp):
                timestamp = pd.to_datetime(timestamp)
                
            if timestamp <= current_datetime and rca_data['rca'] != "No active issues detected":
                # Clean up the RCA and recommendations text
                rca_text = rca_data['rca'].replace("1. :\n\n", "").strip()
                recommendations_text = rca_data['recommendations'].strip()
                
                if rca_text and recommendations_text:  # Only add if both have content
                    current_rca_entries.append({
                        'timestamp': timestamp,
                        'rca': rca_text,
                        'recommendations': recommendations_text
                    })

        # If timeline reset, append new entries to existing history
        if was_reset:
            # Keep existing entries and add new ones avoiding duplicates
            existing_entries = {(pd.to_datetime(entry['timestamp']), entry['rca']) 
                            for entry in noc_state.rca_history}
            
            for entry in current_rca_entries:
                entry_key = (entry['timestamp'], entry['rca'])
                if entry_key not in existing_entries:
                    noc_state.rca_history.append(entry)
                    existing_entries.add(entry_key)
        else:
            # Normal update
            noc_state.rca_history.extend(current_rca_entries)

        # Update RCA in NOC state
        current_rca = get_current_rca(rca_by_timestamp, current_datetime)
        if current_rca['rca'] != "No active issues detected":
            noc_state.update_rca(current_rca, current_datetime)

        # Sort RCA history by timestamp (most recent first)
        noc_state.rca_history.sort(key=lambda x: pd.to_datetime(x['timestamp']) if isinstance(x['timestamp'], str) else x['timestamp'], reverse=True)

    except Exception as e:
        print(f"Error processing RCA: {str(e)}")
        if not was_reset:
            noc_state.rca_history = []
        current_rca = {
            'rca': "Error processing RCA",
            'recommendations': "Analysis temporarily unavailable"
        }

    # Prepare state for serialization
    state_dict = {
        'anomalies': noc_state.anomalies,
        'alert_history': [
            {
                'key': alert['key'],
                'alert': alert['alert'],
                'first_seen': alert['first_seen'],
                'resolved': alert['resolved'],
                'resolution_time': alert['resolution_time'],
                'resolution_details': alert['resolution_details'],
                'severity_level': alert.get('severity_level', 1)
            }
            for alert in noc_state.alert_history
        ],
        'resolution_history': noc_state.resolution_history,
        'rca_history': noc_state.rca_history
    }

    # Return updated dashboard components
    return (
        str(current_time),
        current_time,
        html.H4(current_datetime.strftime('%Y-%m-%d %H:%M:%S'), 
                className="mb-0 ms-2 text-primary fw-bold"),
        create_metric_figure('AMF', amf_df, current_datetime, window_anomalies['amf']),
        create_metric_figure('SMF', smf_df, current_datetime, window_anomalies['smf']),
        create_metric_figure('UPF', upf_df, current_datetime, window_anomalies['upf']),
        create_anomaly_figure(noc_state.anomalies),
        create_alerts_display(noc_state, current_datetime),
        create_combined_rca_display(noc_state.rca_history),
        json.dumps(state_dict, cls=CustomJSONEncoder)
    )

# Auto-scroll the RCA container
app.clientside_callback(
    """
    function(n_intervals) {
        const container = document.getElementById('rca-history-container');
        if (container) {
            // Only auto-scroll if already near the bottom
            const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 50;
            if (isNearBottom) {
                container.scrollTop = container.scrollHeight;
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('rca-history-container', 'children'),
    [Input('interval-component', 'n_intervals')],
    prevent_initial_call=True
)

if __name__ == '__main__':
    print("\nEnhanced 5G Core Network Operations Center Starting...")
    print(f"Loading data from {os.getcwd()}/data2 directory")
    print(f"Using device: {accelerator.device}")
    print("\nServer starting on http://127.0.0.1:35004/")
    app.run(host='0.0.0.0', port=35004, debug=False)