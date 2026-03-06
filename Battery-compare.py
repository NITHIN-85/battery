import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================= FILE PATHS =================
LFP_PATH = r"F:\ThinkClock\Lico_analysis\LFP-1.csv"
GAYAM_PATH = r"C:\Users\redde\OneDrive\Desktop\ThinkClock\xcel\single_cell_charge_discharge.csv"

DV = 0.005
ICA_RANGE = [-150,150]

# ================= COLORS =================
COLOR_FIRST = "rgb(255,0,0)"     # red
COLOR_LAST = "rgb(0,255,0)"      # green
COLOR_GAYAM = "rgb(0,0,255)"     # blue
LINE_WIDTH = 3


# ================= FUNCTIONS =================

def load_dataset(path):

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    n_cells = len(df.columns)//4

    return df,n_cells


def capacity_to_soc(Q):

    Qmin = np.min(Q)
    Qmax = np.max(Q)

    if Qmax == Qmin:
        return np.zeros_like(Q)

    return (Q-Qmin)/(Qmax-Qmin)*100


def interpolate_dqdv(V,Q):

    idx = np.argsort(V)

    V = V[idx]
    Q = Q[idx]

    Vn = np.arange(V.min(),V.max(),DV)

    Qi = np.interp(Vn,V,Q)

    # Raw ICA (no smoothing)
    dqdv = np.gradient(Qi,Vn)

    return Vn,dqdv


def extract_cell(df,cell_index):

    c = 4*cell_index

    Vc = df.iloc[:,c].dropna().to_numpy()
    Qc = df.iloc[:,c+1].dropna().to_numpy()

    Vd = df.iloc[:,c+2].dropna().to_numpy()
    Qd = df.iloc[:,c+3].dropna().to_numpy()

    SOCc = capacity_to_soc(Qc)
    SOCd = capacity_to_soc(Qd)

    Vci,dQci = interpolate_dqdv(Vc,Qc)
    Vdi,dQdi = interpolate_dqdv(Vd,Qd)

    return SOCc,Vc,SOCd,Vd,dQci,Vci,dQdi,Vdi


# ================= LOAD DATA =================

lfp_df,lfp_cells = load_dataset(LFP_PATH)
gayam_df,gayam_cells = load_dataset(GAYAM_PATH)

FIRST_CYCLE = 0
LAST_CYCLE = lfp_cells-1


# ================= DASH APP =================

app = Dash(__name__)
app.title = "Battery OCV / ICA Dashboard"


app.layout = html.Div([

html.H2(
"Battery OCV / ICA Dashboard",
style={"textAlign":"center","margin":"5px"}
),

html.Div([

html.Div([
html.Label("Gayam Motors Cell"),
dcc.Dropdown(
id="gayam-cell",
options=[{"label":f"Cell-{i+1}","value":i} for i in range(gayam_cells)],
value=0
)
],style={"width":"40%"}),

html.Div([
html.Label("Mode"),
dcc.Dropdown(
id="mode",
options=[
{"label":"Single","value":"single"},
{"label":"Compare","value":"compare"}
],
value="compare"
)
],style={"width":"40%"})

],
style={
"display":"flex",
"gap":"30px",
"marginBottom":"5px"
}
),

dcc.Graph(
id="battery-graph",
style={
"flex":"1",
"height":"100%"
}
)

],
style={
"margin":"10px",
"height":"100vh",
"display":"flex",
"flexDirection":"column"
}
)


# ================= CALLBACK =================

@app.callback(
Output("battery-graph","figure"),
Input("gayam-cell","value"),
Input("mode","value")
)

def update_graph(gayam_cell,mode):

    fig = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    subplot_titles=[
    "OCV (Voltage vs SOC)",
    "ICA (dQ/dV vs Voltage)"
    ])

    def plot_cell(df,cell,label,color):

        SOCc,Vc,SOCd,Vd,dQci,Vci,dQdi,Vdi = extract_cell(df,cell)

        # Voltage vs SOC
        fig.add_trace(go.Scatter(
        x=SOCc,
        y=Vc,
        name=f"{label} Charge",
        line=dict(color=color,width=LINE_WIDTH)
        ),1,1)

        fig.add_trace(go.Scatter(
        x=SOCd,
        y=Vd,
        name=f"{label} Discharge",
        line=dict(color=color,width=LINE_WIDTH)
        ),1,1)

        # ICA
        fig.add_trace(go.Scatter(
        x=dQci,
        y=Vci,
        line=dict(color=color,width=LINE_WIDTH),
        showlegend=False
        ),1,2)

        fig.add_trace(go.Scatter(
        x=dQdi,
        y=Vdi,
        line=dict(color=color,width=LINE_WIDTH),
        showlegend=False
        ),1,2)


    # ---------- LFP reference ----------
    plot_cell(lfp_df,FIRST_CYCLE,"LFP First Cycle",COLOR_FIRST)
    plot_cell(lfp_df,LAST_CYCLE,"LFP Last Cycle",COLOR_LAST)


    # ---------- Gayam Motors ----------
    if mode=="compare":
        plot_cell(gayam_df,gayam_cell,f"Gayam Cell-{gayam_cell+1}",COLOR_GAYAM)


    fig.update_layout(
    template="plotly_white",
    hovermode="closest",
    legend=dict(
    orientation="v",
    y=1,
    x=1.02
    ),
    margin=dict(l=40,r=40,t=40,b=40)
    )

    fig.update_xaxes(title="SOC (%)",range=[0,100],row=1,col=1)
    fig.update_xaxes(title="dQ/dV (mAh/V)",range=ICA_RANGE,row=1,col=2)

    fig.update_yaxes(title="Voltage (V)")

    return fig


# ================= RUN =================

if __name__=="__main__":
    app.run(debug=True)