from pathlib import Path
import pandas as pd
import streamlit as st


def _to_decimal(series: pd.Series) -> pd.Series:
    """
    Convierte una serie a float interpretando coma como separador decimal.
    Soporta mezcla de enteros ('36829') y decimales ('566,53').
    Si coexisten coma y punto, asume coma decimal y punto como miles → quita puntos.
    """
    # Si ya es numérica, devuélvela como float
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip()
    s = s.replace({"": pd.NA, "None": pd.NA, "nan": pd.NA})
    # Mantén solo dígitos y , . signos
    s = s.str.replace(r"[^\d,.\-+]", "", regex=True)
    # Si hay coma y punto, quita puntos (asumiendo que son separadores de miles)
    both = s.str.contains(",") & s.str.contains(r"\.")
    s = s.mask(both, s.str.replace(".", "", regex=False))
    # Cambia coma decimal -> punto
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def cargar_datos_vulnerabilidad():
    if "df_vulnerabilidad" not in st.session_state:
        ruta_proyecto = Path(__file__).resolve().parent.parent
        ruta_archivo = ruta_proyecto / "db" / "vulnerabilidad.xlsx"

        if not ruta_archivo.exists():
            st.error(f"No se encontró el archivo: {ruta_archivo}")
            st.stop()

        dtype_dict = {
            "identificacion": str,
            "ruc_empleador": str,
            "ced_padre": str,
            "ced_madre": str,
        }

        df = {}
        xls = pd.ExcelFile(ruta_archivo)

        for hoja in xls.sheet_names:
            # Fuerza el nombre de hoja a string (evita .lower() sobre int)
            hoja_key = str(hoja).strip().lower()

            df_hoja = pd.read_excel(
                ruta_archivo,
                sheet_name=hoja,  # <- aquí sigue usando el nombre original
                dtype=dtype_dict,
                keep_default_na=False,
            )

            # Asegura que TODOS los encabezados sean string
            df_hoja.columns = df_hoja.columns.map(lambda c: str(c).strip())

            # SOLO normaliza salario/valor; nada de año/mes
            if hoja_key == "ingresos" and "salario" in df_hoja.columns:
                df_hoja["salario"] = _to_decimal(df_hoja["salario"])

            if hoja_key == "deudas" and "valor" in df_hoja.columns:
                df_hoja["valor"] = _to_decimal(df_hoja["valor"])

            df[str(hoja)] = df_hoja  # guarda con nombre de hoja como string

        st.session_state.df_vulnerabilidad = df

    return st.session_state.df_vulnerabilidad
