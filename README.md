# Flood'd
Insight Data Science project

http://floodd.herokuapp.com

## Motivation
Prediction of flood risk is going to become more complicated with the manifestation of climate change effects. Therefore, I built a model to predict flood risk from physical predictors rather than historical extrapolation. The project then converts flood risk into an interpretable and actionable value for homeowners by returning the expected value of 30 years of flood insurance premiums payments.  

## Data sources
1. [Claims data](https://www.fema.gov/policy-claim-statistics-flood-insurance) from FEMA's National Flood Insurance Program.
2. Coastline under projected [sea level rise](https://coast.noaa.gov/slrdata/) made available by NOAA.
3. [Elevation](https://dwtkns.com/srtm/) from Shuttle Radar Topographical Mission made available by USGS.
3. Housing density by census tract by year from the [American Community Survey](https://factfinder.census.gov/faces/nav/jsf/pages/index.xhtml).
4. [Precipitation](https://www.worldclim.org/cmip5_2.5m#com) projected using the ACCESS 1.0 model under climate change RCP 8.5 scenario.

## Method
### Framework
The goal is to predict the probability of taking flood damage of a certain value in a given year. To acheive this, I created separate models for the probability of flood damage and the probability of a payout value given that damage occurred. 
### Pre-processing
Generated metadata for NFIP claims data by pulling elevation, coastal proximity, housing density, and precipitation for all Florida locations in the NFIP database. Claims data is made available with location data at a resolution of 0.1 degrees (about 11 km). To include predictions for locations without flood claims instances, metadata was compiled over a grid of 0.1 degrees for all of Florida.
1. Elevation was pulled as the corresponding value from a 90-m resolution GeoTIFF. 
2. Coastal proximity was calculated as the nearest neighbor between each location and the nearest location on the coastline shapefile. 
3. Housing density was calculated using the housing number and area of the nearest census tract to a lcoation, as calculated by nearest neighbor to census tracts centroids.
4. Precipitation was pulled from raster files at a resolution of 2.5 minutes (about 4.5 kilometers)
### Modeling
Claims instances were modeled with a Poisson regression. Payout given a claim was modeled with a Random Forest regression.
