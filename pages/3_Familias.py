import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.carga_datos import cargar_datos_vulnerabilidad
from utils.filtros import (
    aplicar_filtros,
)
from utils.tarjetas import tarjeta_simple, COLORES
import numpy as np
from utils.hogarUnico import make_hogar_id


# =========================
# Helpers y constantes
# =========================
EMPLEOS_VALIDOS = ["Relacion de Dependencia", "Afiliacion Voluntaria"]
EMPLEOS_TODOS = EMPLEOS_VALIDOS + ["Desconocido"]


def _ingresos_mes6_map(datos_filtrados) -> dict[str, str]:
    """
    Mapa identificacion -> tipo_empleo (Jun/2025).
    Si no aparece, el tipo se considera 'Desconocido'.
    """
    df_ing = datos_filtrados.get("Ingresos", pd.DataFrame())
    if df_ing.empty:
        return {}
    df = df_ing[(df_ing["anio"] == 2025) & (df_ing["mes"] == 6)][
        ["identificacion", "tipo_empleo"]
    ].copy()
    if df.empty:
        return {}
    df["tipo_empleo"] = df["tipo_empleo"].astype(str).str.strip()
    return dict(zip(df["identificacion"], df["tipo_empleo"]))


def _aplicar_filtros_papas_familia(
    familias_df: pd.DataFrame,
    tipo_map: dict[str, str],
    cant_papas: int | None,
    cant_papas_trab: int | None,
    tipo_empleo_sel: str | None,  # "Todos" o un tipo puntual
) -> pd.DataFrame:
    """
    Agrega columnas auxiliares (n_papas, n_trab, tipo_padre/madre, trabaja_padre/madre)
    y aplica los 3 filtros. Devuelve el DataFrame filtrado.
    """
    if familias_df.empty:
        return familias_df

    df = familias_df.copy()

    # IDs limpios
    df["ced_padre"] = (
        df["ced_padre"].astype(str).str.strip().replace({"": "0", "nan": "0"})
    )
    df["ced_madre"] = (
        df["ced_madre"].astype(str).str.strip().replace({"": "0", "nan": "0"})
    )

    # tipos por padre/madre (Desconocido si no está en Ingresos)
    def tipo_de(emp_id: str) -> str:
        if emp_id == "0":
            return "Desconocido"
        t = tipo_map.get(emp_id, "Desconocido")
        return t if t in EMPLEOS_VALIDOS else "Desconocido"

    df["tipo_padre"] = df["ced_padre"].map(tipo_de)
    df["tipo_madre"] = df["ced_madre"].map(tipo_de)

    # quién trabaja en JUN/2025 (según tipo)
    df["trabaja_padre"] = df["tipo_padre"].isin(EMPLEOS_VALIDOS)
    df["trabaja_madre"] = df["tipo_madre"].isin(EMPLEOS_VALIDOS)

    # cantidad de papás y de papás trabajando
    df["n_papas"] = (df["ced_padre"] != "0").astype(int) + (
        df["ced_madre"] != "0"
    ).astype(int)
    df["n_trab"] = df["trabaja_padre"].astype(int) + df["trabaja_madre"].astype(int)

    # (1) filtro cantidad de papás
    if cant_papas in (1, 2):
        df = df[df["n_papas"] == cant_papas]

    # (2) filtro tipo de empleo seleccionado (única opción)
    if tipo_empleo_sel and tipo_empleo_sel != "Todos":
        df = df[
            (df["tipo_padre"] == tipo_empleo_sel)
            | (df["tipo_madre"] == tipo_empleo_sel)
        ]

    # (3) filtro cantidad de papás trabajando
    if cant_papas_trab in (0, 1, 2):
        df = df[df["n_trab"] == cant_papas_trab]

    return df


def _remove_outliers_iqr(
    df: pd.DataFrame, cols: list[str], k: float = 1.5
) -> pd.DataFrame:
    """
    Filtra outliers por IQR en las columnas indicadas.
    Mantiene filas que estén dentro de [Q1 - k*IQR, Q3 + k*IQR] para *todas* las cols.
    """
    if df.empty:
        return df
    mask = pd.Series([True] * len(df), index=df.index)
    for c in cols:
        if c not in df.columns:
            continue
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        low = q1 - k * iqr
        high = q3 + k * iqr
        mask &= df[c].between(low, high)
    return df[mask]


def contar_hogares_unicos(familias_df: pd.DataFrame) -> int:
    """
    Retorna el número de hogares únicos en el DataFrame.
    Un hogar se define por la combinación (ced_padre, ced_madre).
    """
    if familias_df.empty:
        return 0

    hogar_ids = familias_df.apply(
        lambda r: make_hogar_id(r["ced_padre"], r["ced_madre"]), axis=1
    )
    hogar_ids = hogar_ids[hogar_ids != ""]
    return hogar_ids.nunique()


def mediana_salario_por_hogar(familias_df: pd.DataFrame) -> float:
    """
    Retorna la mediana del salario familiar considerando hogares únicos.
    """
    if familias_df.empty or "salario_familiar" not in familias_df.columns:
        return 0.0

    tmp = familias_df.copy()
    tmp["hogar_id"] = tmp.apply(
        lambda r: make_hogar_id(r["ced_padre"], r["ced_madre"]), axis=1
    )
    tmp = tmp[tmp["hogar_id"] != ""]

    salarios_hogares = tmp.groupby("hogar_id", as_index=False)[
        "salario_familiar"
    ].first()["salario_familiar"]

    return float(salarios_hogares.median()) if not salarios_hogares.empty else 0.0


def obtener_datos_familias(datos_filtrados, periodo, grupo_seleccionado):
    """
    Obtiene los datos de familias combinando información de estudiantes, padres y madres
    """
    # Obtener estudiantes del periodo
    df_personas = datos_filtrados["Personas"]
    estudiantes_periodo = df_personas[df_personas["periodo"] == periodo].copy()

    # Obtener universo de familiares
    df_universo = datos_filtrados.get("Universo Familiares", pd.DataFrame())
    if df_universo.empty:
        return pd.DataFrame()

    # Unir estudiantes con información de familiares
    familias = estudiantes_periodo.merge(df_universo, on="identificacion", how="left")

    # Filtrar solo familias que tienen al menos un padre o madre válido
    familias = familias[(familias["ced_padre"] != "0") | (familias["ced_madre"] != "0")]

    return familias


def obtener_datos_empleabilidad_familia(datos_filtrados, familias_df):
    """
    Obtiene información de empleabilidad (desempleo) de las familias
    """
    # Obtener datos de ingresos para determinar empleabilidad
    df_ingresos = datos_filtrados.get("Ingresos", pd.DataFrame())

    # Inicializar columnas de empleabilidad
    familias_df = familias_df.copy()
    familias_df["padre_desempleado"] = True  # Inicialmente asumir desempleado
    familias_df["madre_desempleado"] = True  # Inicialmente asumir desempleado
    familias_df["familia_con_desempleo"] = False

    if not df_ingresos.empty:
        # Filtrar por año 2025 mes 6
        ingresos_mes6 = df_ingresos[
            (df_ingresos["anio"] == 2025) & (df_ingresos["mes"] == 6)
        ]

        if not ingresos_mes6.empty and "salario" in ingresos_mes6.columns:
            # Obtener lista de personas que SÍ tienen ingresos (están empleados)
            personas_empleadas = ingresos_mes6[ingresos_mes6["salario"] > 0][
                "identificacion"
            ].unique()

            # Verificar empleo por familia
            for idx, familia in familias_df.iterrows():
                # Verificar si el padre está empleado
                if familia["ced_padre"] != "0":
                    familias_df.loc[idx, "padre_desempleado"] = (
                        familia["ced_padre"] not in personas_empleadas
                    )
                else:
                    familias_df.loc[idx, "padre_desempleado"] = False  # No aplica

                # Verificar si la madre está empleada
                if familia["ced_madre"] != "0":
                    familias_df.loc[idx, "madre_desempleado"] = (
                        familia["ced_madre"] not in personas_empleadas
                    )
                else:
                    familias_df.loc[idx, "madre_desempleado"] = False  # No aplica

                # Familia tiene desempleo si al menos un padre/madre válido está desempleado
                tiene_padre_valido = familia["ced_padre"] != "0"
                tiene_madre_valida = familia["ced_madre"] != "0"

                padre_desemp = (
                    tiene_padre_valido and familias_df.loc[idx, "padre_desempleado"]
                )
                madre_desemp = (
                    tiene_madre_valida and familias_df.loc[idx, "madre_desempleado"]
                )

                familias_df.loc[idx, "familia_con_desempleo"] = (
                    padre_desemp or madre_desemp
                )

    return familias_df


def obtener_datos_salario_deuda_familia(datos_filtrados, familias_df):
    """
    Combina salarios y deudas familiares (suma de padre + madre)
    """
    # Obtener datos de salarios e ingresos
    df_ingresos = datos_filtrados.get("Ingresos", pd.DataFrame())
    df_deudas = datos_filtrados.get("Deudas", pd.DataFrame())

    result_familias = familias_df.copy()
    result_familias["salario_familiar"] = 0.0
    result_familias["deuda_familiar"] = 0.0

    if not df_ingresos.empty:
        # Filtrar por año 2025 mes 6
        ingresos_mes6 = df_ingresos[
            (df_ingresos["anio"] == 2025) & (df_ingresos["mes"] == 6)
        ]

        # La columna correcta para ingresos es 'salario'
        if "salario" in ingresos_mes6.columns and not ingresos_mes6.empty:
            # Sumar salarios por familia
            for idx, familia in result_familias.iterrows():
                salario_total = 0.0

                # Salario del padre
                if familia["ced_padre"] != "0":
                    salario_padre = ingresos_mes6[
                        ingresos_mes6["identificacion"] == familia["ced_padre"]
                    ]["salario"].sum()
                    salario_total += float(salario_padre) * 14

                # Salario de la madre
                if familia["ced_madre"] != "0":
                    salario_madre = ingresos_mes6[
                        ingresos_mes6["identificacion"] == familia["ced_madre"]
                    ]["salario"].sum()
                    salario_total += float(salario_madre) * 14

                result_familias.loc[idx, "salario_familiar"] = salario_total

    if not df_deudas.empty:
        # Filtrar por año 2025 mes 6
        deudas_mes6 = df_deudas[(df_deudas["anio"] == 2025) & (df_deudas["mes"] == 6)]

        # La columna correcta para deudas es 'valor'
        if "valor" in deudas_mes6.columns and not deudas_mes6.empty:
            # Sumar deudas por familia
            for idx, familia in result_familias.iterrows():
                deuda_total = 0.0

                # Deuda del padre
                if familia["ced_padre"] != "0":
                    deuda_padre = deudas_mes6[
                        deudas_mes6["identificacion"] == familia["ced_padre"]
                    ]["valor"].sum()
                    deuda_total += float(deuda_padre)

                # Deuda de la madre
                if familia["ced_madre"] != "0":
                    deuda_madre = deudas_mes6[
                        deudas_mes6["identificacion"] == familia["ced_madre"]
                    ]["valor"].sum()
                    deuda_total += float(deuda_madre)

                result_familias.loc[idx, "deuda_familiar"] = deuda_total

    return result_familias


def crear_scatter_salario_deuda(familias_df, periodo, grupo_seleccionado):
    if familias_df.empty:
        return None

    tmp = familias_df.copy()
    tmp["hogar_id"] = tmp.apply(
        lambda r: make_hogar_id(r["ced_padre"], r["ced_madre"]), axis=1
    )
    tmp = tmp[tmp["hogar_id"] != ""]

    # tomar un registro por hogar
    hogares = tmp.groupby("hogar_id", as_index=False)[
        ["salario_familiar", "deuda_familiar"]
    ].first()

    hogares_validos = hogares[
        (hogares["salario_familiar"].notna())
        & (hogares["deuda_familiar"].notna())
        & ((hogares["salario_familiar"] > 0) | (hogares["deuda_familiar"] > 0))
    ]

    if hogares_validos.empty:
        return None

    # 🔹 quitar outliers por IQR (Tukey) en ambas variables
    hogares_clean = _remove_outliers_iqr(
        hogares_validos, ["salario_familiar", "deuda_familiar"], k=1.5
    )

    if hogares_clean.empty:
        return None

    fig = px.scatter(
        hogares_clean,
        x="salario_familiar",
        y="deuda_familiar",
        title=f"Relación Salario vs Deuda Familiar - {grupo_seleccionado} {periodo}",
        labels={
            "salario_familiar": "Salario Familiar (USD)",
            "deuda_familiar": "Deuda Familiar (USD)",
        },
        opacity=0.6,
        color_discrete_sequence=["#1f77b4"],
    )

    fig.update_traces(
        marker=dict(size=8),
        hovertemplate="<b>Hogar</b><br>"
        + "Salario: $%{x:,.0f}<br>"
        + "Deuda: $%{y:,.0f}<extra></extra>",
    )

    fig.update_layout(
        height=500, xaxis=dict(tickformat="$,.0f"), yaxis=dict(tickformat="$,.0f")
    )
    return fig


def calcular_salario_promedio_por_hogar(familias_df: pd.DataFrame) -> float:
    if familias_df.empty or "salario_familiar" not in familias_df.columns:
        return 0.0

    hogares = familias_df.copy()

    # Asegura strings limpios; trata vacío como "0"
    hogares["ced_padre"] = (
        hogares["ced_padre"].astype(str).str.strip().replace({"": "0"})
    )
    hogares["ced_madre"] = (
        hogares["ced_madre"].astype(str).str.strip().replace({"": "0"})
    )

    # ID estable de hogar
    hogares["hogar_id"] = hogares.apply(
        lambda r: make_hogar_id(r["ced_padre"], r["ced_madre"]), axis=1
    )
    hogares = hogares[hogares["hogar_id"] != ""]

    # Un solo registro por hogar y promedio del salario familiar
    salario_promedio_hogar = (
        hogares.groupby("hogar_id", as_index=False)["salario_familiar"]
        .first()["salario_familiar"]
        .mean()
    )

    return float(salario_promedio_hogar) if pd.notnull(salario_promedio_hogar) else 0.0


# =========================
# Página
# =========================
st.set_page_config(
    page_title="Análisis de Familias", page_icon="👨‍👩‍👧‍👦", layout="wide"
)

st.title("👨‍👩‍👧‍👦 Análisis de Familias")

# Cargar datos
df_vulnerabilidad = cargar_datos_vulnerabilidad()


# Filtros personalizados para familias (solo E y A)
def mostrar_filtros_familias(df_personas: pd.DataFrame, key_suffix: str = ""):
    """
    Muestra los filtros solo para Enrollment (E) y Afluentes (A)
    """
    st.header("🔍 Filtros")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Solo Enrollment y Afluentes
        opciones_grupo = {"E": "Enrollment", "A": "Afluentes"}

        grupo_seleccionado = st.selectbox(
            "Grupo de interés:",
            options=list(opciones_grupo.keys()),
            format_func=lambda x: opciones_grupo[x],
            index=0,  # Por defecto "E"
            key=f"grupo_interes_familias_{key_suffix}",
        )

    with col2:
        # Facultad (filtrada según el grupo de interés)
        from utils.filtros import obtener_facultades_por_grupo

        facultades_disponibles = obtener_facultades_por_grupo(
            df_personas, grupo_seleccionado
        )

        facultad_seleccionada = st.selectbox(
            "Facultad:",
            options=facultades_disponibles,
            index=0,  # "Todos"
            key=f"facultad_familias_{key_suffix}",
        )

    with col3:
        # Carrera (depende del grupo y facultad)
        from utils.filtros import obtener_carreras_por_grupo_y_facultad

        carreras_disponibles = obtener_carreras_por_grupo_y_facultad(
            df_personas, grupo_seleccionado, facultad_seleccionada
        )

        carrera_seleccionada = st.selectbox(
            "Carrera:",
            options=carreras_disponibles,
            index=0,  # "Todos"
            key=f"carrera_familias_{key_suffix}",
        )

    return grupo_seleccionado, facultad_seleccionada, carrera_seleccionada


# Mostrar filtros
grupo_seleccionado, facultad_seleccionada, carrera_seleccionada = (
    mostrar_filtros_familias(df_vulnerabilidad["Personas"], key_suffix="familias")
)

# Aplicar filtros generales (grupo/facultad/carrera)
datos_filtrados = aplicar_filtros(
    df_vulnerabilidad, grupo_seleccionado, facultad_seleccionada, carrera_seleccionada
)

# Periodos disponibles
df_personas_filtrado = datos_filtrados["Personas"]
periodos = sorted(df_personas_filtrado["periodo"].dropna().unique().tolist())

# === Filtros específicos SOLO para Enrollment (E) ===
if grupo_seleccionado == "E":
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        cant_papas_opt = st.selectbox(
            "Cantidad de papás en el hogar",
            options=["Todos", 1, 2],
            index=0,
            help="Hogares con 1 o 2 representantes (excluye huérfanos).",
            key="f_cant_papas_familias",
        )
        cant_papas = None if cant_papas_opt == "Todos" else int(cant_papas_opt)

    with fc2:
        cant_papas_trab_opt = st.selectbox(
            "Cantidad de papás trabajando (JUN/2025)",
            options=["Todos", 0, 1, 2],
            index=0,
            help="Se considera 'trabajando' si aparece en Ingresos 2025-06 con Relación de Dependencia o Afiliación Voluntaria.",
            key="f_cant_papas_trab_familias",
        )
        cant_papas_trab = (
            None if cant_papas_trab_opt == "Todos" else int(cant_papas_trab_opt)
        )

    with fc3:
        tipo_empleo_sel = st.selectbox(
            "Tipo de empleo (JUN/2025)",
            options=["Todos"] + EMPLEOS_TODOS,
            index=0,
            help="‘Desconocido’ = no aparece en Ingresos del mes 6/2025.",
            key="f_tipo_empleo_familias",
        )
else:
    # Si es A (Afluentes), no aplicamos estos filtros
    cant_papas = None
    cant_papas_trab = None
    tipo_empleo_sel = "Todos"


if periodos:
    st.subheader(f"📊 Análisis Familiar - {grupo_seleccionado}")

    # Si hay al menos 2 periodos, mostrar en columnas
    if len(periodos) >= 2:
        # Dividir en dos columnas para mostrar los periodos
        col1, col_divider, col2 = st.columns([1, 0.05, 1])

        # --------------------
        # Columna izquierda
        # --------------------
        with col1:
            st.write(f"### {periodos[0]}")

            # Obtener datos de familias para el primer periodo
            familias_df = obtener_datos_familias(
                datos_filtrados, periodos[0], grupo_seleccionado
            )

            # ✅ Aplicar filtros SOLO si es Enrollment
            if grupo_seleccionado == "E" and not familias_df.empty:
                tipo_map = _ingresos_mes6_map(datos_filtrados)
                familias_df = _aplicar_filtros_papas_familia(
                    familias_df, tipo_map, cant_papas, cant_papas_trab, tipo_empleo_sel
                )

            if not familias_df.empty:
                # Procesar datos de empleabilidad
                familias_df = obtener_datos_empleabilidad_familia(
                    datos_filtrados, familias_df
                )

                # Procesar datos de salario y deuda
                familias_df = obtener_datos_salario_deuda_familia(
                    datos_filtrados, familias_df
                )

                # Mostrar métricas para primer periodo
                subcol1, subcol2, subcol3 = st.columns(3)
                with subcol1:
                    num_hogares = contar_hogares_unicos(familias_df)
                    tarjeta_simple(
                        "Total Familias", f"{num_hogares:,}", COLORES["azul"]
                    )
                with subcol2:
                    salario_promedio = calcular_salario_promedio_por_hogar(familias_df)
                    tarjeta_simple(
                        "Ingreso Promedio Anual",
                        f"${salario_promedio:,.0f}",
                        COLORES["morado"],
                    )
                with subcol3:
                    mediana_salario = mediana_salario_por_hogar(familias_df)
                    tarjeta_simple(
                        "Ingreso Anual - Mediano",
                        f"${mediana_salario:,.0f}",
                        COLORES["verde"],
                    )
            else:
                st.info("No hay datos de familias disponibles para este periodo")

        # divisor visual
        with col_divider:
            st.markdown(
                """
            <div style="
                border-left: 2px solid #e0e0e0;
                height: 300px;
                margin: 20px 0;
            "></div>
            """,
                unsafe_allow_html=True,
            )

        # --------------------
        # Columna derecha
        # --------------------
        with col2:
            st.write(f"### {periodos[1]}")

            # Obtener datos de familias para el segundo periodo
            familias_df2 = obtener_datos_familias(
                datos_filtrados, periodos[1], grupo_seleccionado
            )

            # ✅ Aplicar filtros SOLO si es Enrollment
            if grupo_seleccionado == "E" and not familias_df2.empty:
                tipo_map = _ingresos_mes6_map(datos_filtrados)
                familias_df2 = _aplicar_filtros_papas_familia(
                    familias_df2, tipo_map, cant_papas, cant_papas_trab, tipo_empleo_sel
                )

            if not familias_df2.empty:
                # Procesar datos de empleabilidad
                familias_df2 = obtener_datos_empleabilidad_familia(
                    datos_filtrados, familias_df2
                )

                # Procesar datos de salario y deuda
                familias_df2 = obtener_datos_salario_deuda_familia(
                    datos_filtrados, familias_df2
                )

                # Mostrar métricas para segundo periodo
                subcol1, subcol2, subcol3 = st.columns(3)
                with subcol1:
                    num_hogares2 = contar_hogares_unicos(familias_df2)
                    tarjeta_simple(
                        "Total Familias", f"{num_hogares2:,}", COLORES["azul"]
                    )
                with subcol2:
                    salario_promedio2 = calcular_salario_promedio_por_hogar(
                        familias_df2
                    )
                    tarjeta_simple(
                        "Ingreso Promedio Anual",
                        f"${salario_promedio2:,.0f}",
                        COLORES["morado"],
                    )
                with subcol3:
                    mediana_salario2 = mediana_salario_por_hogar(familias_df2)
                    tarjeta_simple(
                        "Ingreso Anual - Mediano",
                        f"${mediana_salario2:,.0f}",
                        COLORES["verde"],
                    )
            else:
                st.info("No hay datos de familias disponibles para este periodo")

        # =========================
        # Scatter plots (ambos periodos)
        # =========================
        st.markdown("---")
        st.subheader("📈 Relación Salario vs Deuda Familiar")

        scatter_col1, scatter_col2 = st.columns(2)

        # Izquierda
        with scatter_col1:
            familias_df_scatter1 = obtener_datos_familias(
                datos_filtrados, periodos[0], grupo_seleccionado
            )

            # ✅ Aplicar filtros SOLO si es Enrollment
            if grupo_seleccionado == "E" and not familias_df_scatter1.empty:
                tipo_map = _ingresos_mes6_map(datos_filtrados)
                familias_df_scatter1 = _aplicar_filtros_papas_familia(
                    familias_df_scatter1,
                    tipo_map,
                    cant_papas,
                    cant_papas_trab,
                    tipo_empleo_sel,
                )

            if not familias_df_scatter1.empty:
                familias_df_scatter1 = obtener_datos_salario_deuda_familia(
                    datos_filtrados, familias_df_scatter1
                )
                fig_scatter1 = crear_scatter_salario_deuda(
                    familias_df_scatter1, periodos[0], grupo_seleccionado
                )
                if fig_scatter1:
                    st.plotly_chart(fig_scatter1, use_container_width=True)
                else:
                    st.info(
                        "No hay datos suficientes para mostrar la relación salario-deuda"
                    )

        # Derecha
        with scatter_col2:
            familias_df_scatter2 = obtener_datos_familias(
                datos_filtrados, periodos[1], grupo_seleccionado
            )

            # ✅ Aplicar filtros SOLO si es Enrollment
            if grupo_seleccionado == "E" and not familias_df_scatter2.empty:
                tipo_map = _ingresos_mes6_map(datos_filtrados)
                familias_df_scatter2 = _aplicar_filtros_papas_familia(
                    familias_df_scatter2,
                    tipo_map,
                    cant_papas,
                    cant_papas_trab,
                    tipo_empleo_sel,
                )

            if not familias_df_scatter2.empty:
                familias_df_scatter2 = obtener_datos_salario_deuda_familia(
                    datos_filtrados, familias_df_scatter2
                )
                fig_scatter2 = crear_scatter_salario_deuda(
                    familias_df_scatter2, periodos[1], grupo_seleccionado
                )
                if fig_scatter2:
                    st.plotly_chart(fig_scatter2, use_container_width=True)
                else:
                    st.info(
                        "No hay datos suficientes para mostrar la relación salario-deuda"
                    )

    else:
        # ==============
        # Un solo periodo
        # ==============
        periodo = periodos[0]
        st.write(f"### {periodo}")

        # Obtener datos de familias
        familias_df = obtener_datos_familias(
            datos_filtrados, periodo, grupo_seleccionado
        )

        # ✅ Aplicar filtros SOLO si es Enrollment
        if grupo_seleccionado == "E" and not familias_df.empty:
            tipo_map = _ingresos_mes6_map(datos_filtrados)
            familias_df = _aplicar_filtros_papas_familia(
                familias_df, tipo_map, cant_papas, cant_papas_trab, tipo_empleo_sel
            )

        if not familias_df.empty:
            # Procesar datos de empleabilidad
            familias_df = obtener_datos_empleabilidad_familia(
                datos_filtrados, familias_df
            )

            # Procesar datos de salario y deuda
            familias_df = obtener_datos_salario_deuda_familia(
                datos_filtrados, familias_df
            )

            # Mostrar métricas generales
            col1, col2, col3 = st.columns(3)

            with col1:
                num_hogares = contar_hogares_unicos(familias_df)
                tarjeta_simple("Total Familias", f"{num_hogares:,}", COLORES["azul"])

            with col2:
                salario_promedio = calcular_salario_promedio_por_hogar(familias_df)
                tarjeta_simple(
                    "Ingreso Promedio Anual",
                    f"${salario_promedio:,.0f}",
                    COLORES["morado"],
                )

            with col3:
                mediana_salario = mediana_salario_por_hogar(familias_df)
                tarjeta_simple(
                    "Ingreso Anual - Mediano",
                    f"${mediana_salario:,.0f}",
                    COLORES["verde"],
                )

            # Scatter plot de salario vs deuda
            st.markdown("---")
            fig_scatter = crear_scatter_salario_deuda(
                familias_df, periodo, grupo_seleccionado
            )
            if fig_scatter:
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info(
                    "No hay datos suficientes para mostrar la relación salario-deuda"
                )
        else:
            st.info("No hay datos de familias disponibles para este periodo")
else:
    st.write("No hay periodos disponibles para este grupo")
