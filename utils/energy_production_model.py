import xarray as xr
import numpy as np
from agrifoodpy.utils.scaling import logistic_scale, linear_scale

def solar_panels(
        datablock,
        farm_percentage,
        items=2732,
        land_type=["Improved grassland", "Semi-natural grassland"],
        new_land_type="Solar Panels",
        solar_panel_capacity=0.3,
        specific_yield=1200,
        ground_coverage_ratio=0.5,
        baseline_flock_size=31016701,
        sheep_stock_density=12,
        t_init=2021,
        timescale=20
        ):
    """Replaces pasture land with solar panels, reducing the amount of food
    production, and increasing the amount of energy produced.
    """

    # timescale = datablock["global_parameters"]["timescale"]
    pctg_baseline = datablock["land"]["baseline"].copy(deep=True)
    pctg = datablock["land"]["percentage_land_use"].copy(deep=True)

    to_spare = pctg_baseline.sel({"aggregate_class":land_type})

    # Spare the specified land type
    delta_spared =  to_spare * farm_percentage
    pctg.loc[{"aggregate_class":land_type}] -= delta_spared

    if new_land_type not in pctg.aggregate_class.values:
        spared_new_class = xr.zeros_like(pctg.isel(aggregate_class=0)).where(np.isfinite(pctg.isel(aggregate_class=0)))
        spared_new_class["aggregate_class"] = new_land_type
        pctg = xr.concat([pctg, spared_new_class], dim="aggregate_class")

    if "aggregate_class" in delta_spared.dims:
        pctg.loc[{"aggregate_class":new_land_type}] += delta_spared.sum(dim="aggregate_class")
    else:
        pctg.loc[{"aggregate_class":new_land_type}] += delta_spared

    unit_logistic = logistic_scale(
        y0=2020,
        y1=t_init,
        y2=t_init + timescale,
        y3=2050,
        c_init=0,
        c_end=1
    )

    area_solar_farms_ha_arr = pctg.sel({"aggregate_class": "Solar Panels"}).sum(
    ).values * unit_logistic
    area_solar_panels_ha_arr = area_solar_farms_ha_arr * ground_coverage_ratio
    
    datablock["area_solar_farms_ha_arr"] = area_solar_farms_ha_arr
    datablock["area_solar_panels_ha_arr"] = area_solar_panels_ha_arr
    datablock["installed_maximum_capacity_arr"] = 10000 * area_solar_panels_ha_arr * solar_panel_capacity
    datablock["energy_production_wh_arr"] = area_solar_panels_ha_arr * 10000 * solar_panel_capacity * specific_yield
    
    # Recompute emissions balance

    emissions_sector = datablock["balanced_pathway"]

    baseline_electricity_emissions = xr.DataArray(
        data = [43.77, 32.44, 36.74, 40.80, 47.38, 53.27, 56.20],
        coords={"Year": [2020, 2025, 2030, 2035, 2040, 2045, 2050]},
        name="Electricity"
    ).interp(Year=emissions_sector.Year.values)

    BP_solar_capacity = xr.DataArray(
        data = [16.24, 37.84, 70.01, 82.15, 94.27, 106.4],
        coords={"Year": [2020, 2030, 2035, 2040, 2045, 2050]},
        name="Solar panel capacity"
    ).interp(Year=emissions_sector.Year.values)

    BP_total_capacity = xr.DataArray(
        data = [119.0, 197., 269., 311., 352., 394.],
        coords={"Year": [2020, 2030, 2035, 2040, 2045, 2050]},
        name="Total capacity"
    ).interp(Year=emissions_sector.Year.values)

    delta_solar_capacity = BP_solar_capacity - datablock["installed_maximum_capacity_arr"]/1e9 - 16.24
    delta_solar_capacity = delta_solar_capacity.where(delta_solar_capacity > 0, 0)
    
    delta_emissions = emissions_sector.sel(Sector="Electricity supply") - baseline_electricity_emissions
    added_emissions = delta_solar_capacity / BP_total_capacity * delta_emissions
    
    BP_gas_capacity = xr.DataArray(
        data = [34.88, 33.69, 29.71, 22.90, 13.00, 0.0],
        coords={"Year": [2020, 2030, 2035, 2040, 2045, 2050]},
        name="Gas capacity"
    ).interp(Year=emissions_sector.Year.values)

    emissions_sector.loc[{"Sector":"Electricity supply"}] -= added_emissions
    datablock["balanced_pathway"] = emissions_sector
    
    # Add spared class to the land use map
    datablock["land"]["percentage_land_use"] = pctg

    # Scale food production and imports
    total_sheep_not_produced = delta_spared.sum().values * sheep_stock_density

    new_production_fraction = 1 - total_sheep_not_produced / baseline_flock_size

    food_orig = datablock["food"]["g/cap/day"]
    scale_spare = logistic_food_supply(food_orig, t_init, timescale, 1, new_production_fraction)

    # scaled_items = food_orig.sel(Item=food_orig.Item_origin=="Vegetal Products").Item.values
    scaled_items = get_items(food_orig, items)

    out = food_orig.fbs.scale_add(element_in="production",
                                  element_out="imports",
                                  scale=scale_spare,
                                  items=scaled_items,
                                  add=False)

    ratio = out / food_orig
    ratio = ratio.where(~np.isnan(ratio), 1)
    datablock["food"]["g/cap/day"] = out

    return datablock


def logistic_food_supply(
        fbs,
        t_init,
        timescale,
        c_init,
        c_end
        ):
    """Creates a logistic curve using the year range of the input food balance
    supply"""

    y0 = fbs.Year.values[0]
    y1 = t_init
    y2 = t_init + timescale
    y3 = fbs.Year.values[-1]

    scale = logistic_scale(y0, y1, y2, y3, c_init=c_init, c_end=c_end)

    return scale

def get_items(
        fbs,
        items
        ):
    """Get items from food data."""
    if isinstance(items, tuple):
        items = fbs.sel(Item=np.isin(fbs[items[0]], items[1])).Item.values
    elif np.isscalar(items):
        items = [items]
    return items

def energy_balance_sheet(datablock):
    """Creates an energy balance sheet with the same structure as the food balance
    sheet, but with energy production and imports instead of food production
    and imports."""

    energy_ds = xr.Dataset(
        data_vars={
            "production": (("Year", "Item"),    np.array([[77,	    0,	    33298,	0,	    29566,	14193,	17713,	0,	    0]])),
            "imports": (("Year", "Item"),       np.array([[1163,	967,	52407,	35065,	38977,	6581,	0,	    3760,	0]])),
            "exports": (("Year", "Item"),       np.array([[726,	    5,	    30944,	20741,	10183,	590,	0,	    888,	0]])),
            "demand": (("Year", "Item"),        np.array([[1446,	1097,	55413,	11640,	58578,	20178,	17713,	2928,	0]])),
        },
        coords={
            "Year": [2024],
            "Item": [
                "Coal",
                "Manufactured fuel",
                "Primary oils",
                "Petroleum products",
                "Natural gas",
                "Bioenergy & waste",
                "Primary electricity",
                "Electricity",
                "Heat sold"
                ],
        }
    )

    # Convert from mteo to TWh
    energy_ds *= 0.01163

    energy_ds = energy_ds.reindex(Year=list(range(2020, 2051)), fill_value=None)
    energy_ds = energy_ds.fillna(energy_ds.sel(Year=2024))

    datablock["energy"] = energy_ds

    return datablock