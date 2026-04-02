import streamlit as st
from millify import millify
from utils.altair_plots import plot_bars_altair2, plot_years_altair
from agrifoodpy.utils.scaling import logistic_scale
import numpy as np
import altair as alt
import pandas as pd

def energy_production(datablock):
    cols = st.columns((2,1))

    # Extract relevant data from the datablock
    pctg = datablock["land"]["percentage_land_use"]
    fbs = datablock["food"]["g/cap/day"]
    fbs_baseline = datablock["food"]["baseline"]

    metric_yr = st.session_state["plots_year"]

    # Logistic adoption curve
    unit_logistic = logistic_scale(
        y0=2020,
        y1=st.session_state["t_init_solar_panels"],
        y2=st.session_state["t_init_solar_panels"] + st.session_state["solar_panels_timescale"],
        y3=2050,
        c_init=0,
        c_end=1
    )

    meat_items = fbs.sel(Item=fbs.Item_group == "Meat").Item.values
    area_solar_farms_ha_arr = pctg.sel({"aggregate_class": "Solar Panels"}).sum(
    ).values * unit_logistic

    area_solar_farms_ha = area_solar_farms_ha_arr.sel({"Year":metric_yr})

    area_solar_panels_ha_arr = unit_logistic * area_solar_farms_ha * st.session_state["ground_coverage_ratio"]/100
    area_solar_panels_ha = area_solar_panels_ha_arr.sel({"Year":metric_yr})

    sheep_fbs = fbs.sel({"Item":[2732]}).sel(Year=metric_yr)
    sheep_fbs_baseline = fbs_baseline.sel({"Item":[2732]}).isel(Year=-1)
    sheep_ssr = sheep_fbs.fbs.SSR()
    sheep_ssr_baseline = sheep_fbs_baseline.fbs.SSR()

    meat_fbs = fbs.sel({"Item":meat_items}).isel(Year=-1)
    meat_fbs_baseline = fbs_baseline.sel({"Item":meat_items}).isel(Year=-1)
    meat_ssr =  meat_fbs.fbs.SSR()
    meat_ssr_baseline = meat_fbs_baseline.fbs.SSR()

    installed_maximum_capacity_arr = 10000* area_solar_panels_ha_arr * st.session_state['solar_panel_capacity']
    installed_maximum_capacity = installed_maximum_capacity_arr.sel({"Year":metric_yr})

    energy_production_wh_arr = area_solar_panels_ha_arr * 10000 * st.session_state["solar_panel_capacity"] * st.session_state["specific_yield"]
    energy_production_wh = energy_production_wh_arr.sel({"Year":metric_yr})

    energy_production_ds = datablock["energy"].copy(deep=True)

    energy_production_ds["production"].loc[{"Item":"Primary electricity"}] += energy_production_wh_arr / 1e12 # Convert from Wh to TWh
    energy_production_ds["demand"].loc[{"Item":"Primary electricity"}] += energy_production_wh_arr / 1e12 # Convert from Wh to TWh
    
    energy_production_ds["demand"].loc[{"Item":"Natural gas"}] -= energy_production_wh_arr / 1e12 # Convert from Wh to TWh
    energy_production_ds["imports"].loc[{"Item":"Natural gas"}] -= energy_production_wh_arr / 1e12 # Convert from Wh to TWh

    energy_ssr = energy_production_ds.fbs.SSR().sel(Year=metric_yr)
    energy_ssr_baseline = energy_production_ds.fbs.SSR().sel(Year=2025)

    # -----
    # plots
    # ----- 

    prefixes = ["k", "M", "G", "T", "P", "E", "Z"]

    with cols[0]:
        cols_left, cols_right = st.columns(2, border=True)
        
        with cols_left:
            st.markdown("**Total energy production**")

            st.metric(
                "Installed maximum capacity",
                value=f"{millify(installed_maximum_capacity, precision=2, prefixes=prefixes)} Wp",
                help="Maximum theoretical power the solar panels can produce, under ideal conditions"
            )

            st.metric(
                "Total energy produced from solar farms in pasture land",
                value = f"{millify(energy_production_wh, precision=2, prefixes=prefixes)} Wh",
                help="Total energy produced in Wh per year"
                )
            
        with cols_right:

            st.markdown("**Land use change**")

            st.metric(
                "Total area of pasture land converted to solar farms",
                value= f"{millify(area_solar_farms_ha, precision=2, prefixes=prefixes)} ha",
                help="Total area of pasture land converted to solar farms, in hectares"
            )

            st.metric(
                "Total effective area covered in solar panels",
                value=f"{millify(area_solar_panels_ha, precision=2, prefixes=prefixes)} ha",
                help="Total area of solar panels, in hectares"
            )

        with st.container(border=True, height=500):

            data_dict = {
                "Converted pasture land": area_solar_farms_ha_arr,
                "Total energy produced": energy_production_wh_arr,
                "Installed maximum capacity": installed_maximum_capacity_arr,
                "Area covered in solar panels": area_solar_panels_ha_arr
            }

            metric_to_plot = st.selectbox(
                "Select metric to plot",
                options=list(data_dict.keys()),
            )

            to_plot = data_dict[metric_to_plot]

            to_plot = to_plot.expand_dims({"Item": [metric_to_plot]})
            to_plot.name = metric_to_plot

            c = plot_years_altair(
                to_plot,
                ylabel=metric_to_plot
            ).properties(height=350)
            
            rules = alt.Chart(pd.DataFrame({
                'Year': [metric_yr]
            })).mark_rule().encode(
                x='Year:N',
            )
            
            st.altair_chart(c+rules)
 
    with cols[1]:
        with st.container(border=True):
            fbs_to_chart = st.selectbox(
                "Select food balance element to plot",
                options=["Sheep production", "Energy production"]
            )

            if fbs_to_chart == "Sheep production":
                st.markdown("**Impact on food production**")

                for var in list(sheep_fbs.data_vars):
                    sheep_fbs = sheep_fbs.rename({var:var.capitalize()})

                sheep_fbs = sheep_fbs.rename({"Food": "Retail"})
                sheep_fbs = sheep_fbs.fbs.group_sum(coordinate="Item_name", new_name="Item")            

                sheep_production_chart = plot_bars_altair2(
                    sheep_fbs,
                    show="Item",
                    data_vars=["Production", "Imports"],
                    reversed_vars=["Exports", "Retail"],
                    x_axis_title="g/cap/day",
                )

                st.altair_chart(sheep_production_chart)

                cols_ssr = st.columns(3)

                with cols_ssr[0]:
                    st.metric(
                        "Sheep meat SSR",
                        value = "{:.2f} %".format(100*sheep_ssr),
                        delta = "{:.2f} %".format(100*(sheep_ssr-sheep_ssr_baseline))
                    )

                with cols_ssr[1]:

                    st.metric(
                        "Meat products SSR",
                        value = "{:.2f} %".format(100*meat_ssr),
                        delta = "{:.2f} %".format(100*(meat_ssr-meat_ssr_baseline))
                    )

                with cols_ssr[2]:
                    pass
            
            elif fbs_to_chart == "Energy production":
                st.markdown("**Energy production**")

                to_plot = energy_production_ds.sel(Year=metric_yr)

                energy_production_chart = plot_bars_altair2(
                    to_plot,
                    show="Item",
                    data_vars=["production", "imports"],
                    reversed_vars=["exports", "demand"],
                    x_axis_title="Energy production [TWh]",
                )

                st.altair_chart(energy_production_chart)

                cols_ssr = st.columns(3)

                with cols_ssr[0]:
                    st.metric(
                        "Energy SSR",
                        value = "{:.2f} %".format(100*energy_ssr),
                        delta = "{:.2f} %".format(100*(energy_ssr-energy_ssr_baseline))
                    )
