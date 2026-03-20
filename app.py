"""
Skill Demand and Supply - Poland 2025
Streamlit Dashboard with Datawrapper Visualizations
"""

import streamlit as st
import pandas as pd
import requests
import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg2

# Load environment variables
load_dotenv()

DATAWRAPPER_API_KEY = os.getenv("DATAWRAPPER_API_KEY")
DATAWRAPPER_BASE_URL = "https://api.datawrapper.de/v3"

# Database credentials
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "trainings_pl")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Page configuration
st.set_page_config(
    page_title="Skill Demand & Supply - Poland 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Playfair+Display:wght@600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #f8fafc 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #f8fafc 100%);
    }
    
    h1 {
        font-family: 'Playfair Display', serif !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    h2, h3 {
        font-family: 'DM Sans', sans-serif !important;
        color: #334155 !important;
        font-weight: 500 !important;
    }
    
    p, span, div {
        font-family: 'DM Sans', sans-serif !important;
        color: #475569 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #ffffff;
        border-radius: 12px;
        padding: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 8px;
        color: #64748b;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        padding: 0 24px;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #6366f1;
        font-family: 'DM Sans', sans-serif;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 8px;
    }
    
    .annotation {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.15) 0%, rgba(251, 191, 36, 0.08) 100%);
        border-left: 3px solid #f59e0b;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 16px 0;
        color: #92400e;
        font-size: 0.95rem;
    }
    
    .chart-container {
        background: #ffffff;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .stSpinner > div {
        border-color: #8b5cf6 !important;
    }
    
    /* Style selectbox */
    .stSelectbox [data-baseweb="select"] {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


class DatawrapperAPI:
    """Helper class for Datawrapper API interactions"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = DATAWRAPPER_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def create_chart(self, title: str, chart_type: str = "d3-bars") -> dict:
        """Create a new chart"""
        response = requests.post(
            f"{self.base_url}/charts",
            headers=self.headers,
            json={"title": title, "type": chart_type}
        )
        response.raise_for_status()
        return response.json()
    
    def upload_data(self, chart_id: str, csv_data: str) -> bool:
        """Upload CSV data to a chart"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "text/csv"
        }
        response = requests.put(
            f"{self.base_url}/charts/{chart_id}/data",
            headers=headers,
            data=csv_data
        )
        return response.status_code == 204
    
    def update_properties(self, chart_id: str, properties: dict) -> dict:
        """Update chart properties"""
        response = requests.patch(
            f"{self.base_url}/charts/{chart_id}",
            headers=self.headers,
            json=properties
        )
        response.raise_for_status()
        return response.json()
    
    def publish(self, chart_id: str) -> dict:
        """Publish a chart"""
        response = requests.post(
            f"{self.base_url}/charts/{chart_id}/publish",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_embed_url(self, chart_id: str) -> str:
        """Get embed URL for a published chart"""
        return f"https://datawrapper.dwcdn.net/{chart_id}/"


@st.cache_data(ttl=3600)
def load_jobads_data():
    """Load job ads data from parquet"""
    parquet_path = Path(__file__).parent / "jobads" / "data" / "combined.parquet"
    if not parquet_path.exists():
        st.error(f"Parquet file not found: {parquet_path}")
        return None
    
    df = pd.read_parquet(parquet_path)
    df['month'] = pd.to_datetime(df['posted_date']).dt.to_period('M')
    return df


@st.cache_data(ttl=3600)
def get_jobads_monthly_stats():
    """Get job ads statistics by month"""
    df = load_jobads_data()
    if df is None:
        return None
    
    monthly = df.groupby('month').size().reset_index(name='count')
    monthly['month'] = monthly['month'].astype(str)
    monthly['month_name'] = pd.to_datetime(monthly['month']).dt.strftime('%B')
    return monthly


@st.cache_data(ttl=3600)
def get_available_years():
    """Get list of years available in the trainings data"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        query = """
            SELECT DISTINCT EXTRACT(YEAR FROM data_rozpoczecia_uslugi) as year
            FROM public.bur_services
            WHERE data_rozpoczecia_uslugi IS NOT NULL
            ORDER BY year
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        return [int(year) for year in df['year'].tolist()] if not df.empty else []
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return []


@st.cache_data(ttl=3600)
def load_trainings_data(year: int):
    """Load trainings data from PostgreSQL for a specific year"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        # Get trainings by month for the selected year
        query = """
            SELECT 
                DATE_TRUNC('month', data_rozpoczecia_uslugi) as month,
                COUNT(*) as count
            FROM public.bur_services
            WHERE EXTRACT(YEAR FROM data_rozpoczecia_uslugi) = %s
            GROUP BY DATE_TRUNC('month', data_rozpoczecia_uslugi)
            ORDER BY month
        """
        
        df = pd.read_sql(query, conn, params=(year,))
        conn.close()
        
        if not df.empty:
            df['month'] = pd.to_datetime(df['month']).dt.strftime('%Y-%m')
            df['month_name'] = pd.to_datetime(df['month']).dt.strftime('%B')
        
        return df
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None


@st.cache_data(ttl=3600)
def get_total_trainings(year: int):
    """Get total number of trainings for a specific year"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        query = """
            SELECT COUNT(*) as total
            FROM public.bur_services
            WHERE EXTRACT(YEAR FROM data_rozpoczecia_uslugi) = %s
        """
        
        result = pd.read_sql(query, conn, params=(year,))
        conn.close()
        
        return result['total'].iloc[0] if not result.empty else 0
    except Exception as e:
        return 0


@st.cache_resource
def create_datawrapper_chart(title: str, data_csv: str, chart_type: str, 
                             custom_colors: dict = None, description: str = None,
                             notes: str = None):
    """Create and publish a Datawrapper chart, returning embed URL"""
    if not DATAWRAPPER_API_KEY:
        return None
    
    try:
        dw = DatawrapperAPI(DATAWRAPPER_API_KEY)
        
        # Create chart
        chart = dw.create_chart(title, chart_type)
        chart_id = chart['id']
        
        # Upload data
        dw.upload_data(chart_id, data_csv)
        
        # Update properties
        properties = {
            "metadata": {
                "visualize": {
                    "thick": True,
                    "show-color-key": False,
                },
                "describe": {
                    "source-name": "CBOP / BUR API",
                    "byline": "World Bank",
                }
            }
        }
        
        if custom_colors:
            properties["metadata"]["visualize"]["custom-colors"] = custom_colors
        
        if description:
            properties["metadata"]["describe"]["intro"] = description
            
        if notes:
            properties["metadata"]["annotate"] = {"notes": notes}
        
        dw.update_properties(chart_id, properties)
        
        # Publish
        dw.publish(chart_id)
        
        return dw.get_embed_url(chart_id)
    
    except Exception as e:
        st.error(f"Datawrapper API error: {e}")
        return None


def display_metric_card(value: str, label: str):
    """Display a styled metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def main():
    # Header
    st.markdown("<h1 style='text-align: center; font-size: 2.8rem; margin-bottom: 0.5rem; color: #0f172a !important;'>📊 Skill Demand & Supply</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #6366f1 !important; font-size: 1.3rem; margin-top: 0;'>Poland 2025</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2 = st.tabs(["💼 Job Ads", "🎓 Trainings"])
    
    # Job Ads Tab
    with tab1:
        st.markdown("### Job Advertisements by Month (2025)")
        st.markdown("*Data from Central Job Offers Portal (CBOP)*")
        
        monthly_stats = get_jobads_monthly_stats()
        
        if monthly_stats is not None:
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            total_jobs = monthly_stats['count'].sum()
            avg_jobs = monthly_stats['count'].mean()
            peak_month = monthly_stats.loc[monthly_stats['count'].idxmax()]
            lowest_month = monthly_stats.loc[monthly_stats['count'].idxmin()]
            
            with col1:
                display_metric_card(f"{total_jobs:,}", "Total Job Ads")
            with col2:
                display_metric_card(f"{avg_jobs:,.0f}", "Monthly Average")
            with col3:
                display_metric_card(f"{peak_month['count']:,}", f"Peak ({peak_month['month_name']})")
            with col4:
                display_metric_card(f"{lowest_month['count']:,}", f"Lowest ({lowest_month['month_name']})")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Datawrapper chart
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Prepare CSV data for Datawrapper
            chart_data = monthly_stats[['month', 'count']].copy()
            chart_data.columns = ['Month', 'Job Advertisements']
            csv_data = chart_data.to_csv(index=False, sep=';')
            
            with st.spinner("Creating visualization..."):
                embed_url = create_datawrapper_chart(
                    title="Job Advertisements in Poland - 2025",
                    data_csv=csv_data,
                    chart_type="d3-bars",
                    custom_colors={"Job Advertisements": "#6366f1"},
                    description="Monthly count of job advertisements posted in Poland during 2025."
                )
                
                if embed_url:
                    # Embed the Datawrapper chart
                    st.components.v1.iframe(embed_url, height=450, scrolling=False)
                else:
                    # Fallback to native Streamlit chart
                    st.warning("Datawrapper API key not configured. Showing fallback chart.")
                    st.bar_chart(chart_data.set_index('Month'))
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Data table
            with st.expander("📋 View Raw Data"):
                st.dataframe(
                    monthly_stats[['month', 'month_name', 'count']].rename(
                        columns={'month': 'Month', 'month_name': 'Month Name', 'count': 'Count'}
                    ),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.error("Could not load job ads data. Please check the parquet file exists.")
    
    # Trainings Tab
    with tab2:
        st.markdown("### Training Services by Month")
        st.markdown("*Data from Baza Usług Rozwojowych (BUR)*")
        
        # Get available years
        available_years = get_available_years()
        
        if available_years:
            # Year selector
            col1, col2 = st.columns([1, 3])
            with col1:
                selected_year = st.selectbox(
                    "Select Year",
                    options=available_years,
                    index=0 if 2015 not in available_years else available_years.index(2015)
                )
            
            # Show annotation for 2015 data
            if selected_year == 2015:
                st.markdown("""
                <div class="annotation">
                    ⚠️ <strong>Note:</strong> 2015 data is incomplete. The BUR service was initiated during this year, 
                    so the data represents only a partial year starting from the service launch date.
                </div>
                """, unsafe_allow_html=True)
            
            trainings_data = load_trainings_data(selected_year)
            total_trainings = get_total_trainings(selected_year)
            
            if trainings_data is not None and not trainings_data.empty:
                # Metrics row
                col1, col2, col3 = st.columns(3)
                
                avg_trainings = trainings_data['count'].mean()
                peak_training = trainings_data.loc[trainings_data['count'].idxmax()]
                
                with col1:
                    display_metric_card(f"{total_trainings:,}", "Total Trainings")
                with col2:
                    display_metric_card(f"{avg_trainings:,.0f}", "Monthly Average")
                with col3:
                    display_metric_card(f"{peak_training['count']:,}", f"Peak ({peak_training['month_name']})")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Datawrapper chart
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                
                # Prepare CSV data for Datawrapper
                chart_data = trainings_data[['month', 'count']].copy()
                chart_data.columns = ['Month', 'Training Services']
                csv_data = chart_data.to_csv(index=False, sep=';')
                
                notes = None
                if selected_year == 2015:
                    notes = "Note: 2015 data is incomplete. The BUR service was initiated during this year."
                
                with st.spinner("Creating visualization..."):
                    embed_url = create_datawrapper_chart(
                        title=f"Training Services in Poland - {selected_year}",
                        data_csv=csv_data,
                        chart_type="d3-bars",
                        custom_colors={"Training Services": "#10b981"},
                        description=f"Monthly count of training services registered in Poland during {selected_year}.",
                        notes=notes
                    )
                    
                    if embed_url:
                        # Embed the Datawrapper chart
                        st.components.v1.iframe(embed_url, height=450, scrolling=False)
                    else:
                        # Fallback to native Streamlit chart
                        st.warning("Datawrapper API key not configured. Showing fallback chart.")
                        st.bar_chart(chart_data.set_index('Month'))
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Data table
                with st.expander("📋 View Raw Data"):
                    st.dataframe(
                        trainings_data[['month', 'month_name', 'count']].rename(
                            columns={'month': 'Month', 'month_name': 'Month Name', 'count': 'Count'}
                        ),
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.warning(f"No training data found for {selected_year}. Please ensure the database is connected and contains data.")
                st.info("To populate the database, run: `python trainings/scraper_trainings_bur.py`")
        else:
            st.warning("No training data found. Please ensure the database is connected and contains data.")
            st.info("To populate the database, run: `python trainings/scraper_trainings_bur.py`")
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #64748b; font-size: 0.8rem; padding: 20px;'>
        Data sources: CBOP (Central Job Offers Portal) • BUR (Baza Usług Rozwojowych)<br>
        Dashboard powered by Streamlit & Datawrapper
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

