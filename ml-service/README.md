## Machine Learning Microservice for App

Note : Have python 3.8 installed on your system and well configured
This is in the case you need to run the service locally
Although we will make sure this is as "deploy and forget" as possible

1. Install the depenedencies?

```bash
pip install -r requirements.txt
```

2. Run the Flask API

```bash
python main.py
```

## API Reference

#### Get health checkup

```http
  GET /
```

Returns status of the service

#### Get Crop Recommendation

```http
  POST /predict-crop
```

| Parameter     | Type     | Description                               |
| :------------ | :------- | :---------------------------------------- |
| `nitrogen`    | `number` | **Required**. Nitrogen content of soil    |
| `phosphorus`  | `number` | **Required**. Phosphorus content of soil  |
| `potassium`   | `number` | **Required**. Potassium content of soil   |
| `temperature` | `number` | **Required**. Temperature content of soil |
| `rainfall`    | `number` | **Required**. Rainfall content of soil    |
| `humidity`    | `number` | **Required**. Humidity content of soil    |
| `ph`          | `number` | **Required**. pH content of soil          |

Returns
| Parameter | Type | Description |
| :------------ | :------- | :---------------------------------------- |
| `crop` | `string` | **Required**. Name of the recommended crop |
| `status` | `enum` | **Required**. "SUCCESS" or "ERROR" |

```http
   POST /recommend-fertilizer
```

| Parameter    | Type     | Description                              |
| :----------- | :------- | :--------------------------------------- |
| `nitrogen`   | `number` | **Required**. Nitrogen content of soil   |
| `phosphorus` | `number` | **Required**. Phosphorus content of soil |
| `potassium`  | `number` | **Required**. Potassium content of soil  |
