import streamlit as st
from millify import millify
from utils.altair_plots import plot_bars_altair
import numpy as np

def energy_production(datablock):
    cols = st.columns(3, border=True)

    pctg = datablock["land"]["percentage_land_use"]
    fbs = datablock["food"]["g/cap/day"]
    fbs_baseline = datablock["food"]["baseline"]

    meat_items = fbs.sel(Item=fbs.Item_group == "Meat").Item.values
    area_solar_farms_ha = pctg.sel({"aggregate_class": "Solar Panels"}).sum(
    ).values 

    area_solar_panels_ha = area_solar_farms_ha * st.session_state["ground_coverage_ratio"]/100

    sheep_fbs = fbs.sel({"Item":[2732]}).isel(Year=-1)
    sheep_fbs_baseline = fbs_baseline.sel({"Item":[2732]}).isel(Year=-1)
    sheep_ssr = sheep_fbs.fbs.SSR()
    sheep_ssr_baseline = sheep_fbs_baseline.fbs.SSR()

    meat_fbs = fbs.sel({"Item":meat_items}).isel(Year=-1)
    meat_fbs_baseline = fbs_baseline.sel({"Item":meat_items}).isel(Year=-1)
    meat_ssr =  meat_fbs.fbs.SSR()
    meat_ssr_baseline = meat_fbs_baseline.fbs.SSR()

    installed_maximum_capacity = 10000* area_solar_panels_ha * st.session_state['solar_panel_capacity']

    # area * efficiency
    energy_production_wh = (
        area_solar_panels_ha *
        10000 *
        st.session_state["solar_panel_capacity"] *
        st.session_state["specific_yield"]
    )

    prefixes = ["k", "M", "G", "T", "P", "E", "Z"]

    with cols[0]:
        st.markdown("**Total energy production**")

        st.metric(
            "Installed maximum capacity",
            value=f"{millify(installed_maximum_capacity, precision=2, prefixes=prefixes)} Wp",
            help="Maximum theoretical power the solar panels can produce, under ideal conditions"
        )

        st.metric(
            "Total energy produced from solar farms in pasture land",
            value = f"{millify(energy_production_wh, precision=2)} Wh",
            help="Total energy produced in Wh per year"
            )
        
    with cols[1]:

        st.markdown("**Land use change**")

        st.metric(
            "Total area of pasture land converted to solar farms",
            value= f"{millify(area_solar_farms_ha, precision=2)} ha",
            help="Total area of pasture land converted to solar farms, in hectares"
        )

        st.metric(
            "Total effective area covered in solar panels",
            value=f"{millify(area_solar_panels_ha, precision=2)} ha",
            help="Total area of solar panels, in hectares"
        )
 
    with cols[2]:
        st.markdown("**Impact on food production**")

        for var in list(sheep_fbs.data_vars):
            sheep_fbs = sheep_fbs.rename({var:var.capitalize()})

        sheep_fbs = sheep_fbs.rename({"Food": "Retail"})

        sheep_production_chart = plot_bars_altair(
            sheep_fbs,
            show="Item",
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