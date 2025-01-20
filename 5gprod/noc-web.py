# Author: Fatih E. NAR
# 2025 Texas US
#
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
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
warnings.filterwarnings('ignore')

print("Starting 5G Core Network Operations Center...")

class NOCState:
    def __init__(self):
        self.cumulative_anomalies = {'AMF': 0, 'SMF': 0, 'UPF': 0}  # Add this line
        self.anomalies = {'AMF': 0, 'SMF': 0, 'UPF': 0}
        self.alert_history = []
        self.recommendations_history = []
        self.resolution_history = []
        self.processed_anomalies = set() 

    def update_anomaly_counts(self, component, timestamp, anomaly_id):
        # Create a unique identifier for this anomaly
        anomaly_key = f"{component}_{timestamp}_{anomaly_id}"
        
        # Only increment if we haven't seen this anomaly before
        if anomaly_key not in self.processed_anomalies:
            self.cumulative_anomalies[component] += 1
            self.processed_anomalies.add(anomaly_key)
        
    def update_alerts(self, alert, current_datetime):
        alert_key = f"{alert['component']}_{alert['type']}_{alert['start_time']}_{alert['end_time']}"
        if alert_key not in [a['key'] for a in self.alert_history]:
            self.alert_history.append({
                'key': alert_key,
                'alert': alert,
                'first_seen': current_datetime,
                'resolved': False,
                'resolution_time': None,
                'resolution_details': None
            })
            
    def update_recommendations(self, rca_result, current_datetime):
        if rca_result['recommendations'] != "No current recommendations":
            # Ensure timestamp is datetime
            timestamp = pd.to_datetime(current_datetime)
            
            rec_entry = {
                'timestamp': timestamp,
                'rca': rca_result.get('rca', 'No RCA available'),
                'recommendations': rca_result['recommendations'],
                'related_alerts': [a['alert'] for a in self.alert_history if not a['resolved']]
            }
            # Check if recommendation already exists
            exists = any(
                pd.to_datetime(r['timestamp']) == timestamp and 
                r['recommendations'] == rec_entry['recommendations']
                for r in self.recommendations_history
            )
            if not exists:
                self.recommendations_history.append(rec_entry)

    def resolve_alert(self, alert_key, current_datetime, resolution_details):
        for alert in self.alert_history:
            if alert['key'] == alert_key and not alert['resolved']:
                alert['resolved'] = True
                alert['resolution_time'] = current_datetime
                alert['resolution_details'] = resolution_details
                self.resolution_history.append({
                    'alert': alert['alert'],
                    'resolution_time': current_datetime,
                    'resolution_details': resolution_details
                })

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
                    'resolution_details': alert['resolution_details']
                }
                for alert in state_dict.get('alert_history', [])
            ]
            state.recommendations_history = state_dict.get('recommendations_history', [])
            state.resolution_history = state_dict.get('resolution_history', [])
        return state

# Initialize the Dash app
app = dash.Dash(__name__, 
    external_stylesheets=[
        'https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css'
    ]
)

# Data loading functions
def load_data():
    print(f"Loading data from {os.getcwd()}/data2 directory...")
    amf_df = pd.read_csv('data2/amf_metrics.csv')
    smf_df = pd.read_csv('data2/smf_metrics.csv')
    upf_df = pd.read_csv('data2/upf_metrics.csv')
    
    # Print column names for debugging
    print("AMF columns:", amf_df.columns.tolist())
    print("SMF columns:", smf_df.columns.tolist())
    print("UPF columns:", upf_df.columns.tolist())
    
    for df in [amf_df, smf_df, upf_df]:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
    
    return amf_df, smf_df, upf_df

def load_alerts():
    print("Loading alerts data...")
    with open('data2/alerts.json', 'r') as f:
        return json.load(f)

def load_preprocessed_data():
    print("Loading preprocessed data...")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Load vector stores
    try:
        vector_stores = {}
        for component in ['amf', 'smf', 'upf']:
            vector_stores[component] = FAISS.load_local(
                f"processed_data/{component}_vector_store",
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Successfully loaded {component} vector store")
            
            with open(f"processed_data/{component}_documents.pkl", 'rb') as f:
                docs = pickle.load(f)
                print(f"Loaded {len(docs)} documents for {component}")
    except Exception as e:
        print(f"Warning: Could not load vector stores: {str(e)}")
        vector_stores = {}
    
    # Load anomalies
    try:
        with open('processed_data/anomalies.pkl', 'rb') as f:
            anomalies = pickle.load(f)
        print("Successfully loaded anomalies")
        for component in ['amf', 'smf', 'upf']:
            anomaly_count = len([e for e in anomalies[component] if e['anomalies']])
            print(f"Found {anomaly_count} anomalies for {component}")
    except Exception as e:
        print(f"Warning: Could not load anomalies: {str(e)}")
        anomalies = {'amf': [], 'smf': [], 'upf': []}
    
    # Load RCA results
    try:
        with open('processed_data/rca_results.pkl', 'rb') as f:
            rca_results = pickle.load(f)
            
        rca_by_timestamp = {}
        for result in rca_results:
            timestamp = pd.to_datetime(result['timestamp']) if isinstance(result['timestamp'], str) else result['timestamp']
            rca_by_timestamp[timestamp] = {
                'rca': result['rca'],
                'recommendations': result['recommendations']
            }
        print(f"Successfully loaded {len(rca_results)} RCA results")
    except Exception as e:
        print(f"Warning: Could not load RCA results: {str(e)}")
        rca_by_timestamp = {}
    
    return vector_stores, anomalies, rca_by_timestamp

# Helper functions
def get_window_anomalies(anomalies_list, current_datetime, window_minutes=30):
    window_start = current_datetime - timedelta(minutes=window_minutes)
    window_end = current_datetime + timedelta(minutes=1)
    return [
        entry for entry in anomalies_list
        if window_start <= pd.Timestamp(entry['timestamp']) <= window_end
    ]

def get_severity_color(severity):
    return {
        'CRITICAL': 'danger',
        'MAJOR': 'warning',
        'MINOR': 'info'
    }.get(severity, 'secondary')

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

# Visualization functions
def create_metric_figure(component, df, current_datetime, anomalies_info=None):
    window_start = current_datetime - timedelta(minutes=30)
    df_window = df[(df['timestamp'] >= window_start) & (df['timestamp'] <= current_datetime)]
    
    fig = go.Figure()
    
    # Updated metrics mapping to match actual CSV columns
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
            # Add main metric line
            fig.add_trace(go.Scatter(
                x=df_window['timestamp'],
                y=df_window[metric],
                name=name,
                line=dict(color=color),
                mode='lines+markers',
                marker=dict(size=6, symbol='circle')
            ))
            
            # Add anomaly markers if available
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
                            f"Value: {v['value']:.2f}<br>"
                            f"Mean: {v['mean']:.2f}<br>"
                            f"Std Dev: {v['std']:.2f}<br>"
                            f"Deviation: {v.get('deviation_sigma', 0):.2f}σ<br>"
                            f"Method: {v.get('detection_method', 'unknown')}"
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
        title=dict(text=f'{component} Metrics', x=0.01, y=0.95),
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
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            showline=True,
            linewidth=1,
            linecolor='rgba(128,128,128,0.2)'
        )
    )
    
    return fig.to_dict()

def create_alerts_display(noc_state, current_datetime):
    all_alerts = sorted(noc_state.alert_history, 
                       key=lambda x: datetime.strptime(x['alert']['start_time'], '%Y-%m-%d %H:%M:%S.%f'),
                       reverse=True)
    
    if not all_alerts:
        return html.Div([
            html.P("No alerts", className="mb-2"),
            html.Small("No alerts recorded", className="text-muted")
        ])
    
    alert_items = []
    for alert in all_alerts:
        severity = alert['alert'].get('severity', 'MINOR')
        status_class = "text-danger" if not alert['resolved'] else "text-success"
        
        alert_items.append(
            html.Li([
                html.Div([
                    html.Span(
                        f"{alert['alert']['component']} - {alert['alert']['type']}",
                        className=f"badge bg-{get_severity_color(severity)} me-2"
                    ),
                    html.Small(
                        f"Started: {datetime.strptime(alert['alert']['start_time'], '%Y-%m-%d %H:%M:%S.%f').strftime('%H:%M:%S')}",
                        className="text-muted me-2"
                    ),
                    html.Small(
                        f"Status: {'ACTIVE' if not alert['resolved'] else 'RESOLVED'}",
                        className=f"{status_class} fw-bold"
                    )
                ]),
                html.Div(alert['alert']['description'], className="mt-1")
            ], className="mb-3")
        )
    
    return html.Div([
        html.Ul(alert_items, className="list-unstyled mb-2"),
        html.Small(
            f"Total alerts: {len(noc_state.alert_history)} | "
            f"Active: {len([a for a in noc_state.alert_history if not a['resolved']])} | "
            f"Resolved: {len(noc_state.resolution_history)}",
            className="text-muted"
        )
    ])

def create_anomaly_figure(anomaly_data):
    if not anomaly_data:
        return go.Figure().to_dict()

    df = pd.DataFrame.from_dict(
        anomaly_data, 
        orient='index',
        columns=['Count']
    ).reset_index()
    df.columns = ['Component', 'Count']
    
    colors = {'AMF': '#1f77b4', 'SMF': '#ff7f0e', 'UPF': '#9467bd'}
    fig = go.Figure()
    
    for component in ['AMF', 'SMF', 'UPF']:
        count = df[df['Component'] == component]['Count'].values[0] if component in df['Component'].values else 0
        fig.add_trace(go.Bar(
            x=[component],
            y=[count],
            name=component,
            marker_color=colors.get(component, '#333333'),
            text=[f"Total: {count}"],  # Updated text
            textposition='auto',
        ))
    
    fig.update_layout(
        title=dict(text='Cumulative Anomalies by Component', x=0.5, y=0.95),
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
        xaxis_title="",
        yaxis_title="Total Number of Anomalies",  # Updated title
        plot_bgcolor='rgba(240,240,240,0.5)',
        xaxis=dict(showgrid=False, showline=True, linewidth=1, linecolor='rgba(128,128,128,0.2)'),
        yaxis=dict(
            showgrid=True, 
            gridcolor='rgba(128,128,128,0.2)', 
            showline=True, 
            linewidth=1, 
            linecolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinecolor='rgba(128,128,128,0.2)'
        )
    )
    
    return fig.to_dict()

def create_recommendations_display(noc_state, current_datetime):
    # If we haven't started playing yet, show initial state
    if current_datetime == min_time:
        return html.Div([
            html.P("No recommendations available", className="mb-2"),
            html.Small("Start playback to see recommendations...", className="text-muted")
        ])

    try:
        # Convert all timestamps to datetime objects for sorting
        recommendations_with_time = [
            {
                'timestamp': pd.to_datetime(rec['timestamp']) if isinstance(rec['timestamp'], str) else rec['timestamp'],
                'recommendations': rec['recommendations'],
                'rca': rec.get('rca', 'No RCA available')
            }
            for rec in noc_state.recommendations_history
        ]
        
        # Sort by timestamp
        all_recommendations = sorted(
            recommendations_with_time,
            key=lambda x: x['timestamp'],
            reverse=True
        )
        
        if not all_recommendations:
            return html.Div([
                html.P("No recommendations available", className="mb-2"),
                html.Small("Waiting for analysis...", className="text-muted")
            ])
        
        rec_items = []
        for idx, rec in enumerate(all_recommendations):
            try:
                rec_time = pd.to_datetime(rec['timestamp'])
                rec_items.append(
                    html.Li([
                        html.Div([
                            html.Small(
                                f"#{len(all_recommendations)-idx} - {rec_time.strftime('%H:%M:%S')}",
                                className="text-muted me-2"
                            ),
                            html.Strong("Root Cause:"),
                            html.Div(rec.get('rca', 'No RCA available'), className="mb-2"),
                            html.Strong("Recommendations:"),
                            html.Div(rec['recommendations'])
                        ], className="mt-1")
                    ], className="mb-3 border-bottom pb-2")
                )
            except Exception as e:
                print(f"Error processing recommendation {idx}: {str(e)}")
                continue
        
        return html.Div([
            html.Ul(rec_items, className="list-unstyled mb-2"),
            html.Small(
                f"Total recommendations: {len(all_recommendations)}",
                className="text-muted"
            )
        ])
        
    except Exception as e:
        print(f"Error in create_recommendations_display: {str(e)}")
        return html.Div([
            html.P("Error displaying recommendations", className="mb-2 text-danger"),
            html.Small(str(e), className="text-muted")
        ])

# Create layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("5G Core Network Operations Center", className="text-center text-light bg-dark p-2")
    ]),
    
    # Playback controls
    html.Div([
        html.Div([
            html.Button('Play', id='play-button', className="btn btn-primary me-2"),
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
            html.Div(id='current-time', className="d-inline-block"),
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
            updatemode='drag'
        ),
    ], className="container-fluid p-3 bg-light"),
    
    # Main content
    html.Div([
        # Top row with three time series charts
        html.Div([
            html.Div([
                dcc.Graph(id='amf-metrics-graph')  # Changed ID
            ], className="col-md-4"),
            html.Div([
                dcc.Graph(id='smf-metrics-graph')  # Changed ID
            ], className="col-md-4"),
            html.Div([
                dcc.Graph(id='upf-metrics-graph')  # Changed ID
            ], className="col-md-4"),
        ], className="row mb-4"),

        # Analysis row
        html.Div([
            html.Div([
                html.H4("Anomaly Counts", className="text-center"),
                dcc.Graph(id='anomaly-counts-graph')  # Changed ID
            ], className="col-md-6"),
            html.Div([
                html.H4("Root Cause Analysis", className="text-center"),
                html.Div(id='rca-output', className="alert alert-info overflow-auto", 
                        style={'maxHeight': '300px'})
            ], className="col-md-6"),
        ], className="row mb-4"),

        # Add after the "Analysis row" div
        html.Div([
            html.Div([
                html.H4("Alerts", className="text-center"),
                html.Div(id='alerts-table', className="alert alert-info overflow-auto", 
                        style={'maxHeight': '300px'})
            ], className="col-md-6"),
            html.Div([
                html.H4("Recommendations", className="text-center"),
                html.Div(id='recommendations', className="alert alert-info overflow-auto", 
                        style={'maxHeight': '300px'})
            ], className="col-md-6"),
        ], className="row mb-4"),
        
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
    return 'Pause' if play_state == 'playing' else 'Play'

@app.callback(
    [Output('current-time-store', 'children'),
     Output('time-slider', 'value'),
     Output('current-time', 'children'),
     Output('amf-metrics-graph', 'figure'),
     Output('smf-metrics-graph', 'figure'),
     Output('upf-metrics-graph', 'figure'),
     Output('anomaly-counts-graph', 'figure'),
     Output('alerts-table', 'children'),
     Output('recommendations', 'children'),
     Output('rca-output', 'children'),
     Output('historical-state', 'children')],
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
            'recommendations_history': [],
            'resolution_history': []
        }
        
        return (
            '0',  # current-time-store
            0,    # time-slider
            html.H4(min_time.strftime('%Y-%m-%d %H:%M:%S'), className="mb-0 ms-2"),
            create_metric_figure('AMF', amf_df, min_time, []),
            create_metric_figure('SMF', smf_df, min_time, []),
            create_metric_figure('UPF', upf_df, min_time, []),
            create_anomaly_figure({'AMF': 0, 'SMF': 0, 'UPF': 0}),
            html.P("No alerts", className="mb-0"),
            html.P("No recommendations available", className="mb-0"),
            html.P("Waiting to start analysis...", className="mb-0"),
            json.dumps(empty_state)
        )
    
    # Update current time based on trigger
    current_time = float(current_time)
    if trigger_id == 'interval-component' and play_state == 'playing':
        current_time = current_time + speed
        if current_time >= (max_time - min_time).total_seconds():
            current_time = 0
    elif trigger_id == 'time-slider':
        current_time = slider_value
    
    # Calculate current datetime
    current_datetime = min_time + timedelta(seconds=current_time)
    
    # Initialize or load NOC state
    try:
        noc_state = NOCState.from_dict(json.loads(historical_state)) if historical_state != '{}' else NOCState()
    except:
        noc_state = NOCState()

    # Accumulate historical anomalies up to current time
    amf_anomalies = [entry for entry in anomalies['amf'] 
                     if entry['anomalies'] and pd.Timestamp(entry['timestamp']) <= current_datetime]
    smf_anomalies = [entry for entry in anomalies['smf'] 
                     if entry['anomalies'] and pd.Timestamp(entry['timestamp']) <= current_datetime]
    upf_anomalies = [entry for entry in anomalies['upf'] 
                     if entry['anomalies'] and pd.Timestamp(entry['timestamp']) <= current_datetime]

    # Update accumulated anomaly counts
    noc_state.anomalies = {
        'AMF': len(amf_anomalies),
        'SMF': len(smf_anomalies),
        'UPF': len(upf_anomalies)
    }

    # Update anomaly tracking
    for component, anomalies_list in [('AMF', amf_anomalies), ('SMF', smf_anomalies), ('UPF', upf_anomalies)]:
        for anomaly in anomalies_list:
            timestamp = pd.Timestamp(anomaly['timestamp'])
            # Create a unique identifier based on the anomaly characteristics
            anomaly_id = hash(str(anomaly['anomalies']))  # or some other unique identifier from the anomaly
            noc_state.update_anomaly_counts(component, timestamp, anomaly_id)

    # Use cumulative_anomalies instead of anomalies for the graph
    anomaly_figure = create_anomaly_figure(noc_state.cumulative_anomalies)

    # Get current window anomalies for visualization
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

    # Update alerts in NOC state
    for alert in active_alerts:
        noc_state.update_alerts(alert, current_datetime)

    # Check for resolved alerts
    for alert in noc_state.alert_history:
        if not alert['resolved']:
            alert_end = datetime.strptime(alert['alert']['end_time'], '%Y-%m-%d %H:%M:%S.%f')
            if current_datetime > alert_end:
                noc_state.resolve_alert(
                    alert['key'],
                    current_datetime,
                    "Alert cleared - conditions returned to normal"
                )

    # Get RCA results from preprocessed data
    try:
        with open('processed_data/rca_results.pkl', 'rb') as f:
            rca_results = pickle.load(f)
            
        # Find closest RCA result within 5 minutes
        closest_rca = None
        min_diff = timedelta(minutes=5)
        
        for result in rca_results:
            result_time = pd.Timestamp(result['timestamp'])
            time_diff = abs(result_time - current_datetime)
            if time_diff < min_diff:
                closest_rca = result
                min_diff = time_diff
        
        if closest_rca:
            current_rca = closest_rca
        else:
            current_rca = {
                'rca': "No active issues to analyze",
                'recommendations': "No current recommendations"
            }
    except Exception as e:
        print(f"Error loading RCA results: {str(e)}")
        current_rca = {
            'rca': "Error loading RCA results",
            'recommendations': "No recommendations available"
        }

    # Update recommendations
    noc_state.update_recommendations(current_rca, current_datetime)

    # Prepare state for serialization
    state_dict = {
        'anomalies': noc_state.anomalies,
        'alert_history': [
            {
                'key': alert['key'],
                'alert': alert['alert'],
                'first_seen': alert['first_seen'].isoformat() if isinstance(alert['first_seen'], (datetime, pd.Timestamp)) else alert['first_seen'],
                'resolved': alert['resolved'],
                'resolution_time': alert['resolution_time'].isoformat() if isinstance(alert['resolution_time'], (datetime, pd.Timestamp)) else alert['resolution_time'],
                'resolution_details': alert['resolution_details']
            }
            for alert in noc_state.alert_history
        ],
        'recommendations_history': [
            {
                'timestamp': rec['timestamp'].isoformat() if isinstance(rec['timestamp'], (datetime, pd.Timestamp)) else rec['timestamp'],
                'rca': rec.get('rca', 'No RCA available'),
                'recommendations': rec['recommendations'],
                'related_alerts': rec.get('related_alerts', [])
            }
            for rec in noc_state.recommendations_history
        ],
        'resolution_history': [
            {
                'alert': resolution['alert'],
                'resolution_time': resolution['resolution_time'].isoformat() if isinstance(resolution['resolution_time'], (datetime, pd.Timestamp)) else resolution['resolution_time'],
                'resolution_details': resolution['resolution_details']
            }
            for resolution in noc_state.resolution_history
        ]
    }

    # Create display components
    return (
        str(current_time),
        current_time,
        html.H4(current_datetime.strftime('%Y-%m-%d %H:%M:%S'), className="mb-0 ms-2"),
        create_metric_figure('AMF', amf_df, current_datetime, window_anomalies['amf']),
        create_metric_figure('SMF', smf_df, current_datetime, window_anomalies['smf']),
        create_metric_figure('UPF', upf_df, current_datetime, window_anomalies['upf']),
        create_anomaly_figure(noc_state.anomalies),
        create_alerts_display(noc_state, current_datetime),
        create_recommendations_display(noc_state, current_datetime),
        html.Pre(current_rca['rca'], style={'white-space': 'pre-wrap', 'margin': '0'}),
        json.dumps(state_dict)
    )

if __name__ == '__main__':
    print("Starting 5G Core Network Operations Center...")
    print(f"Loading data from {os.getcwd()}/data directory")
    print("Server starting on http://127.0.0.1:35004/")
    app.run(host='0.0.0.0', port=35004, debug=False)
