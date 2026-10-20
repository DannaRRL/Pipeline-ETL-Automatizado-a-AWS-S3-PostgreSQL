
import io
import os
import boto3
import pandas as pd

from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    PartialCredentialsError
)
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# PRACTICA 5: PIPELINE ETL AWS S3
# ==========================================

REGION = "us-west-2"
BUCKET = "aws-academy-datacenter-bucket-23-02-024"

ORIGEN = "raw/transacciones.csv"
DESTINO = "processed/etl_output.parquet"


def extraer_datos(s3):
    print("\n=== 1. FASE DE EXTRACCION ===")

    respuesta = s3.get_object(
        Bucket=BUCKET,
        Key=ORIGEN
    )

    contenido = respuesta["Body"].read()

    df = pd.read_csv(io.BytesIO(contenido))

    print("Archivo de origen:", ORIGEN)
    print("Registros obtenidos:", len(df))
    print("Columnas:", list(df.columns))
    print("\nPrimeros registros:")
    print(df.head().to_string(index=False))

    return df


def transformar_datos(df):
    print("\n=== 2. FASE DE TRANSFORMACION ===")

    df = df.copy()

    # Verificar que exista la columna monto
    if "monto" not in df.columns:
        raise ValueError(
            "El archivo no contiene la columna 'monto'."
        )

    # Convertir monto a numerico
    df["monto"] = pd.to_numeric(
        df["monto"],
        errors="coerce"
    )

    # Eliminar registros con montos invalidos
    registros_antes = len(df)
    df = df.dropna(subset=["monto"]).copy()

    print(
        "Registros con monto invalido:",
        registros_antes - len(df)
    )

    if df.empty:
        raise ValueError(
            "No existen registros validos para procesar."
        )

    # Discretizacion de montos (binning)
    df["categoria_compra"] = pd.cut(
        df["monto"],
        bins=[float("-inf"), 500, 1500, float("inf")],
        labels=["Baja", "Media", "Alta"]
    )

    # Normalizacion Min-Max
    scaler = MinMaxScaler()

    df["monto_normalizado"] = scaler.fit_transform(
        df[["monto"]]
    )

    # Ingenieria de caracteristicas
    df["score_promocional"] = (
        df["monto_normalizado"] * 100
    )

    # Conversion de fecha
    if "fecha_hora" in df.columns:
        df["fecha_hora"] = pd.to_datetime(
            df["fecha_hora"],
            errors="coerce"
        )

    print("\nDataset transformado:")
    print(df.head(10).to_string(index=False))

    print("\nDistribucion de categorias:")
    print(df["categoria_compra"].value_counts())

    print("\nEstadisticas de normalizacion:")
    print(
        df[["monto", "monto_normalizado",
            "score_promocional"]].describe()
    )

    print("\nRegistros procesados:", len(df))

    return df


def cargar_datos(s3, df):
    print("\n=== 3. FASE DE CARGA ===")

    # Serializacion Parquet en memoria
    buffer = io.BytesIO()

    df.to_parquet(
        buffer,
        engine="pyarrow",
        compression="snappy",
        index=False
    )

    buffer.seek(0)

    tamano = buffer.getbuffer().nbytes

    print("Formato: Apache Parquet")
    print("Compresion: Snappy")
    print("Tamano del archivo:", tamano, "bytes")

    # Subida real a AWS S3
    s3.put_object(
        Bucket=BUCKET,
        Key=DESTINO,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream"
    )

    # Verificar que el archivo exista
    verificacion = s3.head_object(
        Bucket=BUCKET,
        Key=DESTINO
    )

    print("\nArchivo subido correctamente.")
    print("Bucket:", BUCKET)
    print("Ruta:", DESTINO)
    print(
        "Tamano verificado:",
        verificacion["ContentLength"],
        "bytes"
    )


def pipeline_etl_completo():
    print("====================================")
    print(" PRACTICA 5 - PIPELINE ETL AWS S3")
    print("====================================")

    try:
        # Utiliza las credenciales configuradas
        # en el entorno o en ~/.aws/credentials
        s3 = boto3.client(
            "s3",
            region_name=REGION
        )

        df_raw = extraer_datos(s3)
        df_procesado = transformar_datos(df_raw)
        cargar_datos(s3, df_procesado)

        print("\n====================================")
        print("PIPELINE ETL FINALIZADO CORRECTAMENTE")
        print("====================================")

    except (
        NoCredentialsError,
        PartialCredentialsError
    ):
        print(
            "\nERROR: Credenciales AWS no disponibles "
            "o incompletas."
        )

    except ClientError as error:
        codigo = error.response.get(
            "Error", {}
        ).get("Code", "Desconocido")

        print("\nERROR DE AWS:", codigo)
        print(
            "Verifica permisos, bucket, "
            "region y vigencia de credenciales."
        )

    except (BotoCoreError, ValueError, OSError) as error:
        print("\nERROR DEL PIPELINE:", error)


if __name__ == "__main__":
    pipeline_etl_completo()
