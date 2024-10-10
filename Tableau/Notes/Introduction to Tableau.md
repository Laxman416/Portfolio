# Introduction to Tableau <!-- omit in toc -->

It is a data visualisation tool used in data analytics to import and clean data, analyse and visualise data.

**Key Uses:**
- Interactive dashboards
- Real-time data analytics
- Geographical visualisations
- Support for complex data joins and relationships

These notes are adapted from the DataCamp course 'Introduction to Tableau' and provide a comprehensive overview of navigation, terminology, visualization techniques, filtering, calculated fields, and advanced features like reference lines and forecasting.

- [Chapter 1 - Getting Started with Tableau](#chapter-1---getting-started-with-tableau)
  - [Navigating Tableau / Terminology](#navigating-tableau--terminology)
  - [Visualisation](#visualisation)
- [Chapter 2 - Building and Customising Visualisations](#chapter-2---building-and-customising-visualisations)
  - [Filtering and Sorting](#filtering-and-sorting)
  - [Calculated Fields](#calculated-fields)
- [Chapter 3 - Advanced Data Exploration: Mapping, Time Analysis, and Predictive Trends](#chapter-3---advanced-data-exploration-mapping-time-analysis-and-predictive-trends)
  - [Mapping data](#mapping-data)
  - [Working with dates](#working-with-dates)
  - [Reference lines, trend lines and forecasting](#reference-lines-trend-lines-and-forecasting)
- [Chapter 4 - Presenting your Data](#chapter-4---presenting-your-data)
  - [Visually appealing](#visually-appealing)
- [Intro to Dashboards and stories](#intro-to-dashboards-and-stories)

## Chapter 1 - Getting Started with Tableau

In this chapter:
- Load data and workbooks
- Navigate Tableau
- First visualisations

### Navigating Tableau / Terminology

**Data Planes**

- Analytics Pane: Contains tools to apply trend lines, forecasts, and reference lines.
- Data Pane: Displays dimensions and measures from your data source.
  
**Discrete and Measures**

Dataplane shows files loaded, analytics and fields of data source.
- Green fields are continuous
- Blue fields are discrete e.g. Room Type

Position of the fields indicate if fields are treated as dimensions or measures.
- Dimensions: contain Qualitative values like names/dates
- Measures: contain Quantitative values

Can convert fields between continuos and discrete/ dimensions and measures.

Discrete dimension and Continuous measure are most common: eye colour/weight

Discrete measure/ Continuous dimension are less common: shoe size/ date

Dimensions are used to segment data (group together) - average price per room type
Measures can be aggregated

**Canvas**

- Columns: Correspond to the x-axis.
- Rows: Correspond to the y-axis.
- Marks Card: Controls the aesthetic elements like color, size, labels, tooltips, and shape of the chart.

### Visualisation

Show me tab top right that suggests graphs from the data on the canvas.
- easy to try different visualisations, but the type of graph is dependent of the question required to answer.

Common Visualisations:
- Bar charts for categorical data.
- Line charts for time series data.
- Heat maps for showing density or intensity.
- Scatter plots for examining relationships between two variables.

## Chapter 2 - Building and Customising Visualisations

- Sorting and filtering
- Aggregation
- Calculated Fields

### Filtering and Sorting

Filters can be applied at multiple levels, such as:
- Extract filters
- Data source filters
- Context filters - more advanced
- Dimension filters - happens in worksheet
- Measure filters - happens in worksheet
  
Drag to Filter tab to filter fields

**Dimension Filters**

Categorical/discrete data:
- Selecting values: highlight and `keep only`
- Define a pattern with a wildcard: e.g. start with 'T'
- Conditions
- Top/Bottom records

**Measures Filters**

Quantitative data:
- Range of values
- =,>,<, ...
- Null or Non-Null Values

**Sorting**

Default on alphabetical
- Sort by clicking headers or icons on toolbar

### Calculated Fields

Create new data from data that already exists: e.g. speed from time and distance.

- Creates a new field and doesn't manipulate dataset
- Create fields in analysis tab

## Chapter 3 - Advanced Data Exploration: Mapping, Time Analysis, and Predictive Trends

In this chapter:
- Map geographic data
- Work with dates
- Reference lines, trend lines, and forecasting
  
### Mapping data

Two main types of maps:

- Field Maps: Represents geographical regions.
- Symbol Maps: Uses symbols to represent data on a geographical plane.

Drag Country to Canvas -> map with longitude and latitude as Country is geocoded.

Can use size and colour of circles on map to indicate fields.

### Working with dates

Dates are automatically placed in Dimensions Area.

Hierarchy:
- Year
- Quarter
- Month
- Day

Continuous Dates: Useful for creating trends over time (e.g., a time series).

Discrete Dates: Useful for breaking data into distinct periods (e.g., daily transactions).

### Reference lines, trend lines and forecasting

Enhancing visualisation:
- Reference lines: Line drawn on chart representing a value e.g. average
- Trend lines: Line to predict to continuation of a certain trend
- Forecasting: Predicting future values using models.

**Adding the lines**

Under Analytics tab

## Chapter 4 - Presenting your Data

In this chapter:
- Improve and format visualisations
- Convey findings with Dashboards and Stories
  
### Visually appealing

- Legend
- Title (font size)
- Axis labels
- Use some colour
- Synchronies Axis

Format at the workbook and sheet level.
3 types of Sheet:
- Worksheet
- Dashboard
- Story

**Dual Axis Graph**

Drag any measure to the right of the graph - synchronise if necessary

## Intro to Dashboards and stories

- Collection of several views
- Easy to compare data
- Data is automatically connected to your worksheets

Stories are sequence of visualisations to tell a narrative.

**Creating dashboards and stories**

New dashboard button at bottom: drag worksheets onto dashboard.
Can use the legend to float in a dashboard for further customisation.

Filter allows you to filter all the graphs on the dashboard at the same time. For example 'Pokemon' will show Sales only for Pokemon games.
Interaction of dashboard is a powerful tool, such as funnel icon.

