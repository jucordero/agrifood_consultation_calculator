import xarray as xr
import numpy as np
from agrifoodpy.utils.scaling import logistic_scale, linear_scale

def solar_panels(
        datablock,
        farm_percentage,
        items=2732,
        land_type=["Improved grassland", "Semi-natural grassland"],
        new_land_type="Solar Panels",
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