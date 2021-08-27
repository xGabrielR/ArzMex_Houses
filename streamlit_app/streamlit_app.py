import folium
import pandas    as pd
import geopandas as gp
import numpy     as np
import streamlit as st

import shapely.geometry
import folium.plugins as pl
import plotly_express as px

from streamlit_folium import folium_static


st.set_page_config( layout='wide' )

@st.cache( allow_output_mutation=True )
def load_data( data ):
    df = pd.read_csv( data )
    return df

@st.cache( allow_output_mutation=True )
def load_lat( data ):
    datall = pd.read_csv( data )
    return datall

def set_features( data ):

    data = data.drop( columns=['url'] )
    data['lot_m2'] = data['sqft'] / 10.76
    data.columns = ['price', 'address', 'local_area', 'zipcode', 'beds', 'baths', 'sqft', 'lot_m2']
    data['maior_menor'] = data['price'].apply( lambda x: 'maior' if x >= 500000 else 'menor' )
    return data


def set_latlon( dll ):

    dll.columns = ['zipcode', 'lat', 'long']
    return dll


def data_view( data, df1 ):

    html1 = '''<h1 style="color:#7647ff;text-align:center">❏ 𝐕𝐢𝐬𝐮𝐚𝐥𝐢𝐳𝐚𝐜̧𝐚̃𝐨 𝐝𝐨𝐬 𝐃𝐚𝐝𝐨𝐬</h1>'''
    html2 = '''<h2 style="color:#b59cff">Descrição dos Registros</h2>'''
    st.markdown( html1, unsafe_allow_html=True )
    st.markdown( html2, unsafe_allow_html=True )


    st.sidebar.title('Filtros Gerais')
    st.sidebar.write(' ')
    f_mm = st.sidebar.radio( 'Preço maior ou menor que R$ 500,000',
            data['maior_menor'].unique() )


    f_lo = st.sidebar.multiselect( 'Selecione o Local', 
            sorted( set( data['local_area'].unique()  ) ) )

    if f_mm == 'maior':
        data = data.loc[data['maior_menor'] == 'maior']

    else:
        data = data.loc[data['maior_menor'] == 'menor']

    st.write(' ')

    if ( f_lo != [] ):
        data = data.loc[data['local_area'].isin( f_lo ), : ]
        
        if data.empty:
            data = df1.copy()
    else:
        data = data.copy()


    st.write( data.head() )

    return None


def statistic( data ):

    st.sidebar.write(' ')
    st.sidebar.write(' ')

    st.sidebar.title('Filtros Comerciais' )

    c1, c2 = st.columns((1, 2))

    num_att = data.select_dtypes( include=['int64', 'float64'] )
    min_ = pd.DataFrame( num_att.apply( min ) )
    max_ = pd.DataFrame( num_att.apply( max ) )
    mean = pd.DataFrame( num_att.apply( np.mean ) )
    medi = pd.DataFrame( num_att.apply( np.median ) )
    std  = pd.DataFrame( num_att.apply( np.std ) )
    ran  = pd.DataFrame( num_att.apply( lambda x: x.max() - x.min() ) )

    dfmet = pd.concat( [max_, min_, mean, medi, std, ran], axis=1 ).reset_index()
    dfmet.columns = ['attributes', 'max', 'min', 'mean', 'median', 'std', 'range']

    html2 = '''<h2 style="color:#b59cff">Métricas Estatísticas</h2>'''
    c2.markdown( html2, unsafe_allow_html=True )
    c2.dataframe( dfmet )

    df00 = data[['lot_m2', 'zipcode']].groupby('zipcode').count()
    df01 = data[['price', 'zipcode']].groupby('zipcode').mean()
    df02 = data[['baths', 'zipcode']].groupby('zipcode').mean()
    df03 = data[['beds', 'zipcode']].groupby('zipcode').mean()

    df04 = pd.merge( df00, df01, on='zipcode', how='inner' )
    df05 = pd.merge( df02, df03, on='zipcode', how='inner')
    dfm  = pd.merge( df04, df05, on='zipcode', how='inner' )

    html2 = '''<h2 style="color:#b59cff">Distribuição por Zipcode</h2>'''
    c1.markdown( html2, unsafe_allow_html=True )
    c1.dataframe( dfm, height=180, width=800 )
    st.write(' ')

    return None


def location_map( df2 ):

    df2['zipcode'] = df2['zipcode'].apply( lambda x: '85938' if x == 'Apache County' else x )
    df2['zipcode'] = df2['zipcode'].astype('int64')

    df2 = pd.merge( df2, dll, on='zipcode', how='inner' )

    html1 = '''<h1 style="color:#7647ff;text-align:center">❐ 𝐋𝐨𝐜𝐚𝐥𝐢𝐳𝐚𝐜̧𝐚̃𝐨 𝐝𝐨𝐬 𝐈𝐦𝐨́𝐯𝐞𝐢𝐬</h1>'''
    st.markdown( html1, unsafe_allow_html=True )

    c3, c4 = st.columns((2, 1))
    info_map = folium.Map( location=[df2['lat'].mean(),
                                     df2['long'].mean()],
                                     default_start_zoom=15 )

    marker_cluster = pl.MarkerCluster().add_to( info_map )
    for name, row in df2.iterrows():
        folium.Marker( [row['lat'], row['long']],
                        icon=folium.Icon(color='red', icon='home', prefix='fa'),
                        popup='Price R$: {0}, Zipcode: {1}, Baths: {2}, Beds: {3}'.format( row['price'],
                                                                                        row['zipcode'],
                                                                                        row['baths'],
                                                                                        row['beds'] ) ).add_to( marker_cluster )

    with c3:
        folium_static( info_map )

    html3 = '''
    <iframe src="https://giphy.com/embed/g4pdANnOggRYDhvjo9" width="480" height="480" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/transparent-g4pdANnOggRYDhvjo9"></a></p>
    '''
    c4.markdown( html3, unsafe_allow_html=True )

    return None


def price_dist( df1, df2 ):
    
    st.write(" ")
    st.write(' ')

    html1 = '''<h1 style="color:#7647ff;text-align:center">❑ 𝐀𝐭𝐫𝐢𝐛𝐮𝐭𝐨𝐬 𝐝𝐨𝐬 𝐈𝐦𝐨́𝐯𝐞𝐢𝐬</h1>'''
    st.markdown( html1, unsafe_allow_html=True )

    f_pr = st.sidebar.slider( 'Selecione o Preço',
            int( df2['price'].min() ), 
            int( df2['price'].max() + 10 ) ) 

    f_lo = st.sidebar.slider( 'Selecione o M2',
            float( df1['lot_m2'].min() ),
            float( df1['lot_m2'].max() + 2 ) )


    df1 = df1.loc[df1['lot_m2'] <= f_lo]
    df2 = df2.loc[df2['price'] <= f_pr]

    df_pr = df2[['price', 'local_area']].groupby('local_area').mean().reset_index()
    df_m2 = df1[['lot_m2', 'price']].groupby('lot_m2').mean().reset_index()

    html2 = '''<h2 style="color:#b59cff">Preço por Região</h2>'''
    st.markdown( html2, unsafe_allow_html=True )

    fig = px.line( df_pr, x='local_area', y='price' )
    fig.update_layout( plot_bgcolor='#0e1117' )
    st.plotly_chart( fig, use_container_width=True )

    #st.header('Preco por Metro Quadrado')
    html2 = '''<h2 style="color:#b59cff">Preço por Metro Quadrado</h2>'''
    st.markdown( html2, unsafe_allow_html=True )

    fig = px.line( df_m2, x='lot_m2', y='price')
    fig.update_layout( plot_bgcolor='#0e1117' )
    st.plotly_chart( fig, use_container_width=True )

    return None



def histograms():
    st.sidebar.write( ' ' )
    c1, c2 = st.columns((2))
    df3 = load_data( c )

    f_ba = st.sidebar.selectbox( 'Selecione o N° de Quartos', 
            sorted( set( df3['baths'].unique() ) ) )

    f_be = st.sidebar.selectbox( 'Selecione o N° de Banheiros', 
            sorted( set( df3['beds'].unique() ) ) )

    df_bed = df3[df3['baths'] <= f_ba]
    df_bat = df3[df3['beds']  <= f_be]

    #c1.header('Numero de Quartos')
    html2 = '''<h2 style="color:#b59cff">Quantidade de Quartos</h2>'''
    c1.markdown( html2, unsafe_allow_html=True )

    fig = px.histogram( df_bed, x='beds', nbins=10 )
    fig.update_layout( plot_bgcolor='#0e1117' )
    c1.plotly_chart( fig, use_container_width=True )

    #c2.header('Numero de Banheiros')
    html2 = '''<h2 style="color:#b59cff">Quantidade de Banheiros</h2>'''
    c2.markdown( html2, unsafe_allow_html=True )
    fig = px.histogram( df_bat, x='baths', nbins=10 )
    fig.update_layout( plot_bgcolor='#0e1117' )
    c2.plotly_chart( fig, use_container_width=True )

    return None


if __name__ == '__main__':
    # Extraction

    c  = '../data/houses.csv'
    cc = '../data/ll.csv'

    data  = load_data( c )
    dll = load_lat( cc )

    # Transformation
    data = set_features( data )

    dll = set_latlon( dll )

    df1 = data.copy()
    df2 = data.copy()

    data_view( data, df1 )
    
    statistic( data )

    location_map( df2 )

    price_dist( df1, df2 )

    histograms()

    # Loading
