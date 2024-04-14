from jinja2 import Environment, FileSystemLoader
from xhtml2pdf import pisa
import pandas as pd
import datetime 
import geopandas as gpd
from shapely.geometry import Point
import geoplot as gplt
import geoplot.crs as gcrs
import matplotlib.pyplot as plt
import mapclassify as mc
import seaborn as sns


# Utility function
def convert_html_to_pdf(source_html, output_filename):
    # open output file for writing (truncated binary)
    result_file = open(output_filename, "w+b")

    # convert HTML to PDF
    pisa_status = pisa.CreatePDF(
            source_html,                # the HTML to convert
            dest=result_file)           # file handle to recieve result

    # close output file
    result_file.close()                 # close output file

    # return True on success and False on errors
    return pisa_status.err

def generate_report_map(date, fire_gdf):
    contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))
    scheme = mc.Quantiles(fire_gdf['bright_ti4'], k=5)
   
    ax = gplt.polyplot(
        contiguous_usa,
        figsize=(12,7),
        zorder=-1,
        linewidth=1,
        projection=gcrs.AlbersEqualArea(),
        edgecolor='white',
        facecolor='lightgray'
    )
    gplt.pointplot(
        fire_gdf,
        hue='bright_ti4', 
        scheme=scheme, 
        projection=gcrs.AlbersEqualArea(),
        cmap='Reds',
        zorder=-1,
        ax = ax,
        extent= contiguous_usa.total_bounds
    )
    ax.set_title(f"Fire Detection in The U.S. on {date}", fontsize=16)
    plt.savefig('.Reporting/fire_plot.png',bbox_inches='tight')

def generate_report_chart(fire_gdf):
    contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))
    sjoin_gdf = gpd.sjoin(contiguous_usa, fire_gdf) #Spatial join Points to polygons
    df_grouped = sjoin_gdf.groupby('state')["index_right"].agg(['count'])
    df_grouped['state'] = df_grouped.index
    # Reorder this data frame
    df_grouped = df_grouped.sort_values(['count'], ascending=False)
    total_number = df_grouped['count'].sum()
    top_state = df_grouped['state'].iloc[0]
    top_number = df_grouped['count'].iloc[0]
    plt.figure(figsize=(12,7)) 
    ax = sns.barplot(
        y="count", 
        x="state", 
        data=df_grouped, 
        errorbar=None, 
        color='red')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    sns.despine(left=True, bottom=True, right=True)
    plt.savefig('./Reporting/fire_chart.png',bbox_inches='tight')

    return total_number,top_state, top_number

date = datetime.date.today()
Map_Key = 'xxxxxxxxxxxxxxxxxxxx' #go to NASA FIRMS to request a map key
API_url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{Map_Key}/VIIRS_SNPP_NRT/USA/1/{date}"
fire_df = pd.read_csv(API_url)
geometry = [Point(xy) for xy in zip(fire_df.longitude, fire_df.latitude)]
fire_df = fire_df.drop(['longitude', 'latitude'], axis=1)
fire_gdf = gpd.GeoDataFrame(fire_df, crs="EPSG:4326", geometry=geometry)
generate_report_map(date, fire_gdf)
total_number,top_state, top_number = generate_report_chart(fire_gdf)

env = Environment(loader=FileSystemLoader(r'Python_Lab\Reporting'))
template = env.get_template("report_layout.html")

template_vars = {"date" : date,
                 "number":total_number,
                 "state": top_state,
                 "number_state": top_number}
html_out = template.render(template_vars)

convert_html_to_pdf(html_out, 'report.pdf')