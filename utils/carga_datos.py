from pathlib import Path
import pandas as pd
import streamlit as st


def cargar_datos_vulnerabilidad():
    if "df_vulnerabilidad" not in st.session_state:
        # Ruta basada en el archivo app.py, que es el punto de entrada
        ruta_proyecto = (
            Path(__file__).resolve().parent.parent
        )  # sube dos niveles desde utils/
        ruta_archivo = ruta_proyecto / "db" / "vulnerabilidad.xlsx"

        if not ruta_archivo.exists():
            st.error(f"No se encontró el archivo: {ruta_archivo}")
            st.stop()

        # Definir las columnas de identificación que deben leerse como texto
        dtype_dict = {
            "identificacion": str,
            "identificacion_persona": str,
            "identificacion_fam": str,
            "ruc_empleador": str,
            "ced_padre": str,
            "ced_madre": str,
        }

        # Leer todas las hojas del Excel
        df = {}
        excel_file = pd.ExcelFile(ruta_archivo)

        for sheet_name in excel_file.sheet_names:
            df[sheet_name] = pd.read_excel(
                ruta_archivo,
                sheet_name=sheet_name,
                dtype=dtype_dict,
                keep_default_na=False,
            )

        st.session_state.df_vulnerabilidad = df

    return st.session_state.df_vulnerabilidad
