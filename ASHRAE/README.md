### 比赛说明  
  评估能效提高的价值可能是具有挑战性的，因为无法真正知道建筑物如果不进行改进将消耗多少能源。我们能做的最好的事情就是建立反事实模型。一旦对建筑物进行了大修，就将新的（较低的）能耗与原始建筑物的模型值进行比较，以计算出改造的节省量。更准确的模型可以支持更好的市场激励措施，并降低融资成本。  
  这项竞赛要求您根据历史使用率和观察到的天气，在四种能源类型之间建立这些反事实模型。该数据集包括来自全球多个不同地点的一千多座建筑物的三年小时计读数。  
    
### 数据表说明
  __building_metadata.csv__  
  site_id  weater_test.csv的外键  
  building_id  train.csv的外键  
  primary_use 基于EnergyStar属性类型定义的建筑物主要活动类别的指示器  
  square_feet 建筑物的总建筑面积  
  year_built  建筑开放年份  
  floor_count 建筑物的楼层数  

  __train.csv__  
  building_id  building_metadata.csv的外键  
  meter 仪表ID码，读为:{ 0:电，1:冷水，2:蒸汽，3:热水}并非每个建筑物都有所有电表类型。  
  timestamp 测量的时间  
  meter_reading 目标变量。能耗，以kWh为单位。  

  __test.csv__  
  row_id 行号  
  building_id 建筑物ID  
  meter 仪表ID码  
  timestamp 测试数据周期的时间戳  

  __weather_train.csv  weater_test.csv__ 气象站的气象数据尽可能靠近站点(site)  
  site_id   
  timestamp 转换为weekday  
  air_temperature  空气温度，摄氏度  
  cloud_coverage  oktas中被云彩覆盖的天空部分  
  dew_temperature 露水温度，摄氏度  
  precip_depth_1_hr 降水深度，单位：毫米  
  sea_level_pressure 海平面压力，单位：毫巴/百帕斯卡  
  wind_direction 风向，指南针方向(0-360)  
  wind_speed 风速，单位：米/秒  

  __sample_submission.csv__ 
  row_id  
  meter_reading 仪表读数，保留小数点后四位  
### train数据  
#### 时间 dates  
  Train: from 2016-01-01 00:00:00 to 2016-12-31 23:00:00  
  Test: from '2017-01-01 00:00:00' to '2018-05-09 07:00:00'  
  MONTHS : [ 1  2  3  4  5  6  7  8  9 10 11 12]  
#### 缺失数据数 Missing data x Column    
  year_built 12113306  
  floor_count 16630052  
  air_temperature 6163  
  cloud_coverage 8734870  
  dew_temperature 9645  
  precip_depth_1_hr 3658528  
  sea_level_pressure 1141174  
  wind_direction 1358553  
  wind_speed 53181  
#### Buildings and sites
  We have 1449 buildings  
  We have 16 sites  
  More information about each site ...  
  Site  0 	observations:  1076662 	Num of buildings:  105  
  Site  1 	observations:  552034 	Num of buildings:  51  
  Site  2 	observations:  2530025 	Num of buildings:  135  
  Site  3 	observations:  2369014 	Num of buildings:  274  
  Site  7 	observations:  359642 	Num of buildings:  15  
  Site  8 	observations:  567915 	Num of buildings:  70  
  Site  11 	observations:  117259 	Num of buildings:  5  
  Site  12 	observations:  314869 	Num of buildings:  36  
  Site  13 	observations:  2711454 	Num of buildings:  154  
  Site  4 	observations:  746664 	Num of buildings:  91  
  Site  5 	observations:  779195 	Num of buildings:  89  
  Site  6 	observations:  667989 	Num of buildings:  44  
  Site  9 	observations:  2678102 	Num of buildings:  124  
  Site  10 	observations:  411313 	Num of buildings:  30  
  Site  14 	observations:  2499502 	Num of buildings:  102  
  Site  15 	observations:  1743966 	Num of buildings:  124  
#### 消耗量前五的建筑
  建筑号   站点
  1099    13
  778     6
  1197    13
  1168    13
  1159    13
#### 可能老的建筑消耗更多
  Buildings built before 1900:  0  
  Buildings built before 2000:  528  
  Buildings built after 2010:  55  
  Buildings built after 2015:  11  
  
  
