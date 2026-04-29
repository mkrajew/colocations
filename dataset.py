import osmnx as ox
import pandas as pd

place = "Warsaw, Poland"

tags = {
    "amenity": ["restaurant", "cafe", "pharmacy", "school", "bank", "atm", "hospital"],
    "shop": ["supermarket", "convenience"],
    "tourism": ["hotel"],
    "highway": ["bus_stop"],
    "railway": ["tram_stop"],
    "leisure": ["park"],
}

print(f"Fetching OSM features for: {place}")
gdf = ox.features_from_place(place, tags)
print(f"  -> Raw features fetched: {len(gdf)}")
print(f"  -> Geometry types:\n{gdf.geom_type.value_counts().to_string()}")

gdf = gdf.to_crs(2180)  # Poland CS92, meters
gdf["point"] = gdf.geometry.representative_point()
gdf["x"] = gdf["point"].x
gdf["y"] = gdf["point"].y

print("\nCoordinate ranges (EPSG:2180, meters):")
print(
    f"  x: {gdf['x'].min():.2f} .. {gdf['x'].max():.2f}  (span {gdf['x'].max() - gdf['x'].min():.2f} m)"
)
print(
    f"  y: {gdf['y'].min():.2f} .. {gdf['y'].max():.2f}  (span {gdf['y'].max() - gdf['y'].min():.2f} m)"
)


def get_feature_type(row):
    for key in ["amenity", "shop", "tourism", "highway", "railway", "leisure"]:
        val = row.get(key)
        if pd.notna(val):
            return f"{key}={val}"
    return None


gdf["feature_type"] = gdf.apply(get_feature_type, axis=1)

unmatched = gdf["feature_type"].isna().sum()
print(f"\nRows without a matched feature_type (dropped): {unmatched}")

events = gdf[["feature_type", "x", "y"]].dropna().reset_index(drop=True)
events["instance_id"] = events.index
events = events[["instance_id", "feature_type", "x", "y"]]

print(f"\nFinal events: {len(events)}")
print(f"Distinct feature types: {events['feature_type'].nunique()}")
print("\nCounts per feature_type:")
print(events["feature_type"].value_counts().to_string())

print("\nCounts per top-level key:")
print(events["feature_type"].str.split("=").str[0].value_counts().to_string())

print("\nSample rows:")
print(events.head(10).to_string(index=False))

duplicate_coords = events.duplicated(subset=["x", "y"]).sum()
print(f"\nRows sharing exact (x,y) with another row: {duplicate_coords}")

out_path = "data/warsaw_osm_events.csv"
events.to_csv(out_path, index=False)
print(f"\nSaved {len(events)} rows to {out_path}")
