import streamlit as st
from millify import millify

def energy_production(datablock):
    cols = st.columns(3, border=True)

    pctg = datablock["land"]["percentage_land_use"]
    area_solar_panels_ha = pctg.sel({"aggregate_class":"Solar Panels"}).sum().values

    # area * efficiency
    energy_production_wh = (
        area_solar_panels_ha *
        10000 *
        st.session_state["solar_panel_efficiency"] *
        365.25 *
        24
    )

    with cols[0]:
        st.markdown("**Total energy production**")

        st.metric(
            "Total area covered in solar panels",
            value= f"{millify(area_solar_panels_ha, precision=2)} ha"
        )

    with cols[1]:
        st.markdown("**Total energy production**")

        st.metric(
            "Energy from solar panels in pasture land",
            value = f"{millify(energy_production_wh, precision=2)} Wh"
            )
        

