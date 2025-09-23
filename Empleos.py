import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.carga_datos import cargar_datos_vulnerabilidad
from utils.filtros import (
    aplicar_filtros,
    mostrar_filtros,
)
from utils.tarjetas import tarjeta_simple, COLORES
from utils.hogarUnico import make_hogar_id


# --- Helpers para rangos de quintil en tarjetas ---
def _fmt_money(v: float) -> str:
    if v is None:
        return "-"
    try:
        return f"${v:,.2f}"
    except Exception:
        return str(v)


EMPLEOS_VALIDOS = ["Relacion de Dependencia", "Afiliacion Voluntaria"]
EMPLEOS_TODOS = EMPLEOS_VALIDOS + ["Desconocido"]


def _normalizar_ids_familia(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ced_padre"] = (
        df["ced_padre"].astype(str).str.strip().replace({"": "0", "nan": "0"})
    )
    df["ced_madre"] = (
        df["ced_madre"].astype(str).str.strip().replace({"": "0", "nan": "0"})
    )
    return df


def _personas_periodo(datos_filtrados, periodo) -> pd.Series:
    df_personas = datos_filtrados["Personas"]
    return df_personas.loc[
        df_personas["periodo"] == periodo, "identificacion"
    ].drop_duplicates()


def _universo_familiares_periodo(datos_filtrados, periodo) -> pd.DataFrame:
    ids = _personas_periodo(datos_filtrados, periodo)
    df_u = datos_filtrados.get("Universo Familiares", pd.DataFrame())
    if df_u.empty:
        return pd.DataFrame()
    u = df_u[df_u["identificacion"].isin(ids)].copy()
    u = _normalizar_ids_familia(u)
    # hogar_id
    u["hogar_id"] = u.apply(
        lambda r: make_hogar_id(r["ced_padre"], r["ced_madre"]), axis=1
    )
    u = u[(u["hogar_id"] != "") & ((u["ced_padre"] != "0") | (u["ced_madre"] != "0"))]
    return u


def _mapa_hogar_familia(u_valid: pd.DataFrame) -> pd.DataFrame:
    pares = []
    for _, r in u_valid.iterrows():
        if r["ced_padre"] != "0":
            pares.append((r["hogar_id"], r["ced_padre"]))
        if r["ced_madre"] != "0":
            pares.append((r["hogar_id"], r["ced_madre"]))
    if not pares:
        return pd.DataFrame(columns=["hogar_id", "fam_id"])
    return pd.DataFrame(pares, columns=["hogar_id", "fam_id"]).drop_duplicates()


def _ingresos_mes6(datos_filtrados) -> pd.DataFrame:
    df_ing = datos_filtrados.get("Ingresos", pd.DataFrame())
    if df_ing.empty:
        return pd.DataFrame(columns=["identificacion", "salario", "tipo_empleo"])
    df = df_ing[(df_ing["anio"] == 2025) & (df_ing["mes"] == 6)].copy()
    # tipificar y sanitizar
    df["tipo_empleo"] = df["tipo_empleo"].astype(str).str.strip()
    df["salario"] = pd.to_numeric(df["salario"], errors="coerce")
    return df[["identificacion", "salario", "tipo_empleo"]]


def construir_enrollment_filtrado(
    datos_filtrados,
    periodo: str,
    cant_papas: int | None,
    cant_papas_trab: int | None,
    tipos_empleo_seleccionados: list[str],
):
    """
    Devuelve un dict con:
      - familiares_ids: set[str]
      - hogares_ids: set[str]
      - df_hogar_salarios: DataFrame con salario total JUN/2025 por hogar (solo filtrados)
      - resumen: dict con counts útiles
    Todos los filtros se aplican aquí.
    """
    u = _universo_familiares_periodo(datos_filtrados, periodo)
    if u.empty:
        return {
            "familiares_ids": set(),
            "hogares_ids": set(),
            "df_hogar_salarios": pd.DataFrame(columns=["hogar_id", "salario"]),
            "resumen": {
                "total_hogares": 0,
                "total_familiares": 0,
                "hogares_por_papas": 0,
                "hogares_por_papas_trab": 0,
            },
        }

    # -- cantidad de papás en hogar (1 o 2)
    u["n_papas"] = (u["ced_padre"].ne("0")).astype(int) + (
        u["ced_madre"].ne("0")
    ).astype(int)
    if cant_papas in (1, 2):
        u = u[u["n_papas"] == cant_papas]
    if u.empty:
        return {
            "familiares_ids": set(),
            "hogares_ids": set(),
            "df_hogar_salarios": pd.DataFrame(columns=["hogar_id", "salario"]),
            "resumen": {
                "total_hogares": 0,
                "total_familiares": 0,
                "hogares_por_papas": 0,
                "hogares_por_papas_trab": 0,
            },
        }

    df_mapa = _mapa_hogar_familia(u)
    if df_mapa.empty:
        return {
            "familiares_ids": set(),
            "hogares_ids": set(),
            "df_hogar_salarios": pd.DataFrame(columns=["hogar_id", "salario"]),
            "resumen": {
                "total_hogares": u["hogar_id"].nunique(),
                "total_familiares": 0,
                "hogares_por_papas": u["hogar_id"].nunique(),
                "hogares_por_papas_trab": 0,
            },
        }

    # Ingresos JUN/2025
    ing6 = _ingresos_mes6(datos_filtrados)

    # Tipo de empleo por familiar en JUN/2025 (Desconocido si no aparece)
    df_emp = df_mapa.merge(
        ing6, left_on="fam_id", right_on="identificacion", how="left"
    )
    df_emp["tipo_empleo_mes6"] = df_emp["tipo_empleo"].where(
        df_emp["tipo_empleo"].isin(EMPLEOS_VALIDOS), "Desconocido"
    )

    # -- filtro por tipo de empleo (sobre personas)
    if tipos_empleo_seleccionados and set(tipos_empleo_seleccionados) != set(
        EMPLEOS_TODOS
    ):
        personas_validas = set(
            df_emp.loc[
                df_emp["tipo_empleo_mes6"].isin(tipos_empleo_seleccionados), "fam_id"
            ]
        )
        df_emp = df_emp[df_emp["fam_id"].isin(personas_validas)]

    # Recalcular hogares/personas después del filtro por tipo
    if df_emp.empty:
        return {
            "familiares_ids": set(),
            "hogares_ids": set(),
            "df_hogar_salarios": pd.DataFrame(columns=["hogar_id", "salario"]),
            "resumen": {
                "total_hogares": 0,
                "total_familiares": 0,
                "hogares_por_papas": 0,
                "hogares_por_papas_trab": 0,
            },
        }

    # ¿Quién trabaja en JUN/2025?
    df_emp["trabaja_mes6"] = df_emp["tipo_empleo_mes6"].isin(EMPLEOS_VALIDOS)

    # Conteos por hogar
    agg = (
        df_emp.groupby("hogar_id")
        .agg(n_papas=("fam_id", "nunique"), n_trab=("trabaja_mes6", "sum"))
        .reset_index()
    )

    # -- filtro por cantidad de papás trabajando (0/1/2)
    if cant_papas_trab in (0, 1, 2):
        agg = agg[agg["n_trab"] == cant_papas_trab]

    if agg.empty:
        return {
            "familiares_ids": set(),
            "hogares_ids": set(),
            "df_hogar_salarios": pd.DataFrame(columns=["hogar_id", "salario"]),
            "resumen": {
                "total_hogares": 0,
                "total_familiares": 0,
                "hogares_por_papas": 0,
                "hogares_por_papas_trab": 0,
            },
        }

    hogares_filtrados = set(agg["hogar_id"])
    df_emp = df_emp[df_emp["hogar_id"].isin(hogares_filtrados)]

    # IDs finales de familiares
    familiares_ids = set(df_emp["fam_id"].unique())

    # Salario del hogar (suma papá+mamá) en JUN/2025 SOLO para los familiares filtrados
    # Ojo: df_emp ya trae 'salario' desde el primer merge con ing6
    if "salario" not in df_emp.columns:
        # fallback defensivo por si algún cambio upstream quitó 'salario'
        df_emp = df_emp.merge(
            ing6[["identificacion", "salario"]],
            left_on="fam_id",
            right_on="identificacion",
            how="left",
            suffixes=("", "_ing"),
        )
        # si hubo colisión, prioriza 'salario' y si no existe, usa 'salario_ing'
        if "salario_ing" in df_emp.columns:
            df_emp["salario"] = df_emp["salario"].fillna(df_emp["salario_ing"])

    df_emp["salario"] = pd.to_numeric(df_emp["salario"], errors="coerce")
    df_hogar_sal = df_emp.groupby("hogar_id", as_index=False)["salario"].sum()

    return {
        "familiares_ids": familiares_ids,
        "hogares_ids": hogares_filtrados,
        "df_hogar_salarios": df_hogar_sal,
        "resumen": {
            "total_hogares": len(hogares_filtrados),
            "total_familiares": len(familiares_ids),
            "hogares_por_papas": u["hogar_id"].nunique(),
            "hogares_por_papas_trab": len(hogares_filtrados),
        },
        # también útil para calcular activos:
        "df_emp": df_emp,  # columnas: hogar_id, fam_id, tipo_empleo_mes6, trabaja_mes6
    }


# Rangos de referencia JUN/2025 a nivel de HOGAR (suma papá+mamá)
RANGOS_QUINTILES_HOGAR = [
    {"Quintil": 1, "Salario_Min": 1.13, "Salario_Max": 642.03},
    {"Quintil": 2, "Salario_Min": 642.03, "Salario_Max": 909.08},
    {"Quintil": 3, "Salario_Min": 909.08, "Salario_Max": 1415.91},
    {"Quintil": 4, "Salario_Min": 1415.91, "Salario_Max": 2491.60},
    {"Quintil": 5, "Salario_Min": 2491.60, "Salario_Max": 20009.99},
]


def rangos_quintiles_hogar_dict() -> dict:
    """
    Devuelve: {"Quintil 1": "$1.13 – $642.03", ...}
    """
    d = {}
    for q in RANGOS_QUINTILES_HOGAR:
        k = f"Quintil {q['Quintil']}"
        d[k] = f"{_fmt_money(q['Salario_Min'])} – {_fmt_money(q['Salario_Max'])}"
    return d


def filtrar_outliers_iqr(
    serie: pd.Series, k: float = 1.5, min_obs: int = 20
) -> pd.Series:
    """
    Filtra outliers por Tukey (k * IQR).
    - k: 1.5 estándar; usa 3 si quieres ser menos agresivo.
    - min_obs: si hay pocos datos, no filtra.
    """
    s = pd.to_numeric(serie, errors="coerce").dropna()
    if s.size < min_obs:
        return s
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return s
    low = q1 - k * iqr
    high = q3 + k * iqr
    return s[(s >= low) & (s <= high)]


def crear_boxplot_salarios_hogares(datos_filtrados, periodo, grupo_seleccionado):
    """
    Crea un boxplot de salarios a nivel de hogar (1 registro por hogar único).
    Si hay hermanos, se considera un único hogar.
    Para el ingreso del hogar se suma el salario del padre y de la madre (si existen).
    """

    if grupo_seleccionado not in ["A", "E"]:
        return None

    # Estudiantes del periodo
    df_personas = datos_filtrados["Personas"]
    estudiantes_periodo = df_personas[df_personas["periodo"] == periodo][
        "identificacion"
    ].drop_duplicates()

    # Universo de familiares
    df_universo = datos_filtrados.get("Universo Familiares", pd.DataFrame())
    if df_universo.empty:
        return None

    # Filtrar universo por estudiantes del periodo
    universo_periodo = df_universo[
        df_universo["identificacion"].isin(estudiantes_periodo)
    ].copy()

    # Normalizar y crear hogar_id único (independiente del orden padre/madre)
    universo_periodo["ced_padre"] = (
        universo_periodo["ced_padre"].astype(str).str.strip().replace({"": "0"})
    )
    universo_periodo["ced_madre"] = (
        universo_periodo["ced_madre"].astype(str).str.strip().replace({"": "0"})
    )
    universo_periodo["hogar_id"] = universo_periodo.apply(
        lambda r: make_hogar_id(r["ced_padre"], r["ced_madre"]), axis=1
    )
    # Quedarse con hogares válidos
    universo_periodo = universo_periodo[universo_periodo["hogar_id"] != ""]
    if universo_periodo.empty:
        return None

    # Traer ingresos (mes 6)
    df_ingresos = datos_filtrados.get("Ingresos", pd.DataFrame())
    if df_ingresos.empty:
        return None
    df_ingresos_mes6 = df_ingresos[
        (df_ingresos["anio"] == 2025) & (df_ingresos["mes"] == 6)
    ]

    # Mapear hogar_id -> integrantes (padre/madre) y unir con ingresos
    pares = []
    for _, r in universo_periodo.iterrows():
        if r["ced_padre"] != "0":
            pares.append((r["hogar_id"], r["ced_padre"]))
        if r["ced_madre"] != "0":
            pares.append((r["hogar_id"], r["ced_madre"]))
    df_mapa = pd.DataFrame(pares, columns=["hogar_id", "fam_id"]).drop_duplicates()
    if df_mapa.empty:
        return None

    df_merge = df_mapa.merge(
        df_ingresos_mes6[["identificacion", "salario"]],
        left_on="fam_id",
        right_on="identificacion",
        how="left",
    )

    # Salario del hogar = suma de salarios de sus integrantes (papá+mamá si aplica)
    df_hogar_salario = df_merge.groupby("hogar_id", as_index=False)["salario"].sum()
    salarios = df_hogar_salario["salario"].dropna()

    # 🔹 quitar outliers automáticamente (Tukey 1.5×IQR)
    salarios = filtrar_outliers_iqr(salarios, k=1.5)

    if salarios.empty:
        return None

    # Boxplot
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=salarios,
            name="Salarios por Hogar",
            boxpoints=False,
            marker_color="lightgreen",
            line_color="darkgreen",
        )
    )
    grupo_nombre = "Afluentes" if grupo_seleccionado == "A" else "Enrollment"
    fig.update_layout(
        title=f"Distribución de Salarios por Hogar - Familiares {grupo_nombre} {periodo}",
        yaxis_title="Salario (USD)",
        height=400,
        showlegend=False,
        yaxis=dict(tickformat="$,.0f"),
    )
    return fig


def calcular_empleo_graduados(datos_filtrados, periodo):
    """
    Calcula empleados activos y desempleados para graduados en un periodo específico

    Args:
        datos_filtrados: Diccionario con los datos filtrados
        periodo: Periodo específico a analizar

    Returns:
        tuple: (activos, desempleados)
    """
    # Obtener graduados únicos del periodo específico (evitar duplicados por múltiples carreras)
    df_personas = datos_filtrados["Personas"]
    graduados_periodo = df_personas[df_personas["periodo"] == periodo][
        "identificacion"
    ].drop_duplicates()

    total_graduados = len(graduados_periodo)

    # Obtener datos de ingresos unificados
    df_ingresos = datos_filtrados.get("Ingresos", pd.DataFrame())

    if df_ingresos.empty:
        return 0, total_graduados

    # Filtrar por anio 2025, mes 6 y graduados del periodo (el último mes para determinar empleo actual)
    df_ingresos_mes6 = df_ingresos[
        (df_ingresos["anio"] == 2025)
        & (df_ingresos["mes"] == 6)
        & (df_ingresos["identificacion"].isin(graduados_periodo))
    ]

    # Obtener identificaciones únicas de graduados que trabajan en mes 6
    graduados_activos = df_ingresos_mes6["identificacion"].drop_duplicates()

    activos = len(graduados_activos)
    desempleados = total_graduados - activos

    return activos, desempleados


def calcular_empleo_familiares(datos_filtrados, periodo):
    """
    Calcula empleados activos y desempleados para familiares (A/E) y métricas por hogar único.

    Returns:
        tuple: (activos, desempleados, total_hogares, hogares_con_trabajo)
            - activos/desempleados: personas familiares únicas (padres/madres)
            - total_hogares: hogares únicos reales (combinación padre|madre)
            - hogares_con_trabajo: hogares con ≥1 familiar activo
    """
    # 1) Estudiantes del periodo
    df_personas = datos_filtrados["Personas"]
    estudiantes_periodo = df_personas[df_personas["periodo"] == periodo][
        "identificacion"
    ].drop_duplicates()

    # 2) Universo de familiares
    df_universo = datos_filtrados.get("Universo Familiares", pd.DataFrame())
    if df_universo.empty:
        return 0, 0, 0, 0

    universo_periodo = df_universo[
        df_universo["identificacion"].isin(estudiantes_periodo)
    ].copy()

    # Normalizar IDs (string) y filtrar filas con al menos un familiar válido
    universo_periodo["ced_padre"] = (
        universo_periodo["ced_padre"].astype(str).str.strip().replace({"": "0"})
    )
    universo_periodo["ced_madre"] = (
        universo_periodo["ced_madre"].astype(str).str.strip().replace({"": "0"})
    )
    u_valid = universo_periodo[
        (universo_periodo["ced_padre"] != "0") | (universo_periodo["ced_madre"] != "0")
    ].copy()

    # 3) Construir hogar_id único (independiente del orden padre/madre)
    u_valid["hogar_id"] = u_valid.apply(
        lambda r: make_hogar_id(r["ced_padre"], r["ced_madre"]), axis=1
    )
    u_valid = u_valid[u_valid["hogar_id"] != ""]

    # Total de hogares únicos reales
    total_hogares = u_valid["hogar_id"].nunique()

    # 4) Familiares (personas) únicos del periodo (padres + madres)
    cedulas_padres = u_valid.loc[
        u_valid["ced_padre"] != "0", "ced_padre"
    ].drop_duplicates()
    cedulas_madres = u_valid.loc[
        u_valid["ced_madre"] != "0", "ced_madre"
    ].drop_duplicates()
    familiares_unicos = pd.concat([cedulas_padres, cedulas_madres]).drop_duplicates()
    total_familiares = len(familiares_unicos)

    # 5) Ingresos (mes 6 de 2025) para empleo actual
    df_ingresos = datos_filtrados.get("Ingresos", pd.DataFrame())
    if df_ingresos.empty:
        # Sin ingresos: nadie activo, hogares_con_trabajo=0
        return 0, total_familiares, total_hogares, 0

    df_ingresos_mes6 = df_ingresos[
        (df_ingresos["anio"] == 2025) & (df_ingresos["mes"] == 6)
    ]

    # Personas familiares activas (únicas) en mes 6 que pertenecen al periodo
    familiares_activos_mes6 = df_ingresos_mes6["identificacion"].drop_duplicates()
    activos_en_periodo = familiares_activos_mes6[
        familiares_activos_mes6.isin(familiares_unicos)
    ]
    activos = len(activos_en_periodo)
    desempleados = total_familiares - activos

    # 6) Hogares con trabajo (≥1 familiar activo) — por hogar único
    # Mapa hogar_id -> fam_id (padre/madre)
    mapa = []
    for _, r in u_valid.iterrows():
        if r["ced_padre"] != "0":
            mapa.append((r["hogar_id"], r["ced_padre"]))
        if r["ced_madre"] != "0":
            mapa.append((r["hogar_id"], r["ced_madre"]))
    df_mapa = pd.DataFrame(mapa, columns=["hogar_id", "fam_id"]).drop_duplicates()

    hogares_con_trabajo = df_mapa[df_mapa["fam_id"].isin(set(activos_en_periodo))][
        "hogar_id"
    ].nunique()

    return activos, desempleados, total_hogares, hogares_con_trabajo


def crear_boxplot_salarios(datos_filtrados, periodo, grupo_seleccionado):
    """
    Crea un boxplot de salarios según el grupo de interés

    Args:
        datos_filtrados: Diccionario con los datos filtrados
        periodo: Periodo específico a analizar
        grupo_seleccionado: Tipo de grupo (G, A, E)

    Returns:
        plotly.graph_objects.Figure: Gráfico boxplot
    """
    if grupo_seleccionado == "G":  # Graduados
        # Obtener graduados del periodo
        df_personas = datos_filtrados["Personas"]
        graduados_periodo = df_personas[df_personas["periodo"] == periodo][
            "identificacion"
        ].drop_duplicates()

        # Obtener ingresos unificados
        df_ingresos = datos_filtrados.get("Ingresos", pd.DataFrame())
        if df_ingresos.empty:
            return None

        # Filtrar por año 2025, mes 6 y graduados del periodo
        df_ingresos_mes6 = df_ingresos[
            (df_ingresos["anio"] == 2025)
            & (df_ingresos["mes"] == 6)
            & (df_ingresos["identificacion"].isin(graduados_periodo))
        ]

        salarios = df_ingresos_mes6["salario"].dropna()
        titulo = f"Distribución de Salarios - Graduados {periodo}"

        # 🔹 quitar outliers automáticamente
        salarios = filtrar_outliers_iqr(salarios, k=1.5)

    else:  # Afluentes o Enrollment
        # Obtener estudiantes del periodo
        df_personas = datos_filtrados["Personas"]
        estudiantes_periodo = df_personas[df_personas["periodo"] == periodo][
            "identificacion"
        ].drop_duplicates()

        # Obtener universo de familiares
        df_universo = datos_filtrados.get("Universo Familiares", pd.DataFrame())
        if df_universo.empty:
            return None

        # Filtrar universo por estudiantes del periodo
        universo_periodo = df_universo[
            df_universo["identificacion"].isin(estudiantes_periodo)
        ]

        # Obtener cédulas únicas de familiares
        cedulas_padres = universo_periodo["ced_padre"][
            universo_periodo["ced_padre"] != "0"
        ].drop_duplicates()
        cedulas_madres = universo_periodo["ced_madre"][
            universo_periodo["ced_madre"] != "0"
        ].drop_duplicates()
        familiares_unicos = pd.concat(
            [cedulas_padres, cedulas_madres]
        ).drop_duplicates()

        # Obtener ingresos unificados
        df_ingresos = datos_filtrados.get("Ingresos", pd.DataFrame())
        if df_ingresos.empty:
            return None

        # Filtrar por año 2025, mes 6 y familiares del periodo
        df_ingresos_mes6 = df_ingresos[
            (df_ingresos["anio"] == 2025)
            & (df_ingresos["mes"] == 6)
            & (df_ingresos["identificacion"].isin(familiares_unicos))
        ]

        salarios = df_ingresos_mes6["salario"].dropna()
        grupo_nombre = "Afluentes" if grupo_seleccionado == "A" else "Enrollment"
        titulo = f"Distribución de Salarios - Familiares {grupo_nombre} {periodo}"

        # 🔹 quitar outliers automáticamente
        salarios = filtrar_outliers_iqr(salarios, k=1.5)

    if len(salarios) == 0:
        return None

    # Crear boxplot
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=salarios,
            name="Salarios",
            boxpoints=False,
            marker_color="lightblue",
            line_color="darkblue",
        )
    )

    fig.update_layout(
        title=titulo,
        yaxis_title="Salario (USD)",
        xaxis_title="",
        height=400,
        showlegend=False,
        yaxis=dict(tickformat="$,.0f"),
    )

    return fig


def calcular_quintiles(datos_filtrados, periodo, grupo_seleccionado):
    """
    Calcula la distribución por quintiles según el grupo de interés

    Args:
        datos_filtrados: Diccionario con los datos filtrados
        periodo: Periodo específico a analizar
        grupo_seleccionado: Tipo de grupo (G, A, E)

    Returns:
        dict: Diccionario con porcentajes por quintil
    """
    if grupo_seleccionado == "G":  # Graduados
        # Obtener graduados del periodo
        df_personas = datos_filtrados["Personas"]
        graduados_periodo = df_personas[df_personas["periodo"] == periodo][
            "identificacion"
        ].drop_duplicates()

        # Obtener ingresos unificados
        df_ingresos = datos_filtrados.get("Ingresos", pd.DataFrame())
        if df_ingresos.empty:
            return {}

        # Filtrar por año 2025, mes 6 y graduados del periodo
        df_ingresos_mes6 = df_ingresos[
            (df_ingresos["anio"] == 2025)
            & (df_ingresos["mes"] == 6)
            & (df_ingresos["identificacion"].isin(graduados_periodo))
        ]

        quintiles_data = df_ingresos_mes6["quintil"].dropna()

    else:  # Afluentes o Enrollment
        # Obtener estudiantes del periodo
        df_personas = datos_filtrados["Personas"]
        estudiantes_periodo = df_personas[df_personas["periodo"] == periodo][
            "identificacion"
        ].drop_duplicates()

        # Obtener universo de familiares
        df_universo = datos_filtrados.get("Universo Familiares", pd.DataFrame())
        if df_universo.empty:
            return {}

        # Filtrar universo por estudiantes del periodo
        universo_periodo = df_universo[
            df_universo["identificacion"].isin(estudiantes_periodo)
        ]

        # Obtener cédulas únicas de familiares
        cedulas_padres = universo_periodo["ced_padre"][
            universo_periodo["ced_padre"] != "0"
        ].drop_duplicates()
        cedulas_madres = universo_periodo["ced_madre"][
            universo_periodo["ced_madre"] != "0"
        ].drop_duplicates()
        familiares_unicos = pd.concat(
            [cedulas_padres, cedulas_madres]
        ).drop_duplicates()

        # Obtener ingresos unificados
        df_ingresos = datos_filtrados.get("Ingresos", pd.DataFrame())
        if df_ingresos.empty:
            return {}

        # Filtrar por año 2025, mes 6 y familiares del periodo
        df_ingresos_mes6 = df_ingresos[
            (df_ingresos["anio"] == 2025)
            & (df_ingresos["mes"] == 6)
            & (df_ingresos["identificacion"].isin(familiares_unicos))
        ]

        quintiles_data = df_ingresos_mes6["quintil"].dropna()

    if len(quintiles_data) == 0:
        return {}

    # Calcular distribución por quintiles
    quintiles_count = quintiles_data.value_counts().sort_index()
    total = quintiles_count.sum()

    # Crear diccionario con porcentajes
    quintiles_dict = {}
    for i in range(1, 6):
        count = quintiles_count.get(i, 0)
        percentage = (count / total * 100) if total > 0 else 0
        quintiles_dict[f"Quintil {i}"] = percentage

    return quintiles_dict


def mostrar_tarjetas_quintiles(quintiles_dict, rangos_dict: dict | None = None):
    """
    Muestra las 5 tarjetas de quintiles con porcentajes y (opcional) el rango
    bajo cada tarjeta en tipografía pequeña.

    Args:
        quintiles_dict: {"Quintil 1": 20.0, ...}
        rangos_dict:    {"Quintil 1": "$1.13 – $642.03", ...}  (opcional)
    """
    colores_quintiles = [
        COLORES["rojo"],  # Q1
        COLORES["naranja"],  # Q2
        COLORES["amarillo"],  # Q3
        COLORES["cyan"],  # Q4
        COLORES["verde"],  # Q5
    ]

    st.write("**📊 Distribución por Quintiles de Ingresos Hogar**")

    if not quintiles_dict:
        st.info("No hay datos de quintiles disponibles para este periodo")
        return

    cols = st.columns(5)

    for i in range(5):
        qkey = f"Quintil {i+1}"
        percentage = quintiles_dict.get(qkey, 0)
        with cols[i]:
            # Tarjeta principal (título + %)
            tarjeta_simple(qkey, f"{percentage:.1f}%", colores_quintiles[i])

            # Rango en texto pequeño (si viene)
            if rangos_dict and qkey in rangos_dict:
                st.markdown(
                    f"""
                    <div style="
                        margin-top: 0px;
                        font-size: 0.78rem;
                        color: #6b7280;
                        text-align:center;">
                        {rangos_dict[qkey]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def crear_diagrama_sankey(datos_filtrados, periodo, grupo_seleccionado):
    """
    Crea un diagrama de Sankey para mostrar transiciones de tipo de empleo entre meses

    Args:
        datos_filtrados: Diccionario con los datos filtrados
        periodo: Periodo específico a analizar
        grupo_seleccionado: Tipo de grupo (G, A, E)

    Returns:
        plotly.graph_objects.Figure: Diagrama de Sankey
    """
    # Obtener personas del periodo específico
    df_personas = datos_filtrados["Personas"]
    personas_periodo = df_personas[df_personas["periodo"] == periodo][
        "identificacion"
    ].drop_duplicates()

    if grupo_seleccionado == "G":  # Graduados
        # Usar directamente las identificaciones de graduados
        identificaciones_relevantes = personas_periodo
        df_ingresos = datos_filtrados.get("Ingresos", pd.DataFrame())
        columna_id = "identificacion"

    else:  # Afluentes o Enrollment
        # Obtener familiares de estudiantes del periodo
        df_universo = datos_filtrados.get("Universo Familiares", pd.DataFrame())
        if df_universo.empty:
            return None

        # Filtrar universo por estudiantes del periodo
        universo_periodo = df_universo[
            df_universo["identificacion"].isin(personas_periodo)
        ]

        # Obtener cédulas únicas de familiares
        cedulas_padres = universo_periodo["ced_padre"][
            universo_periodo["ced_padre"] != "0"
        ].drop_duplicates()
        cedulas_madres = universo_periodo["ced_madre"][
            universo_periodo["ced_madre"] != "0"
        ].drop_duplicates()
        identificaciones_relevantes = pd.concat(
            [cedulas_padres, cedulas_madres]
        ).drop_duplicates()

        df_ingresos = datos_filtrados.get("Ingresos", pd.DataFrame())
        columna_id = "identificacion"

    if df_ingresos.empty or len(identificaciones_relevantes) == 0:
        return None

    # Filtrar ingresos por personas relevantes del periodo
    df_ingresos_periodo = df_ingresos[
        df_ingresos[columna_id].isin(identificaciones_relevantes)
    ]

    if df_ingresos_periodo.empty:
        return None

    # Obtener datos para mes 3 y mes 6
    df_mes3 = df_ingresos_periodo[
        (df_ingresos_periodo["anio"] == 2025) & (df_ingresos_periodo["mes"] == 3)
    ]
    df_mes6 = df_ingresos_periodo[
        (df_ingresos_periodo["anio"] == 2025) & (df_ingresos_periodo["mes"] == 6)
    ]

    # Crear diccionarios de tipo de empleo por persona
    empleo_mes3 = dict(zip(df_mes3[columna_id], df_mes3["tipo_empleo"]))
    empleo_mes6 = dict(zip(df_mes6[columna_id], df_mes6["tipo_empleo"]))

    # Determinar estado de empleo para todas las personas relevantes
    transiciones = []
    for persona in identificaciones_relevantes:
        estado_mes3 = empleo_mes3.get(persona, "Desconocido")
        estado_mes6 = empleo_mes6.get(persona, "Desconocido")
        transiciones.append((estado_mes3, estado_mes6))

    # Contar transiciones
    from collections import Counter

    contador_transiciones = Counter(transiciones)

    # Tipos de empleo posibles
    tipos_empleo = ["Relacion de Dependencia", "Afiliacion Voluntaria", "Desconocido"]

    # Crear nodos del diagrama
    nodos_marzo = [f"{tipo} (Marzo)" for tipo in tipos_empleo]
    nodos_junio = [f"{tipo} (Junio)" for tipo in tipos_empleo]
    todos_nodos = nodos_marzo + nodos_junio

    # Crear mapeo de índices
    indice_nodos = {nodo: i for i, nodo in enumerate(todos_nodos)}

    # Preparar datos para el diagrama de Sankey
    source = []
    target = []
    value = []

    for (origen, destino), cantidad in contador_transiciones.items():
        if cantidad > 0:  # Solo incluir transiciones que existen
            source.append(indice_nodos[f"{origen} (Marzo)"])
            target.append(indice_nodos[f"{destino} (Junio)"])
            value.append(cantidad)

    if not source:  # No hay transiciones para mostrar
        return None

    # Colores para los nodos
    colores_nodos = [
        "#ff6b6b",
        "#4ecdc4",
        "#45b7d1",  # Marzo
        "#ff6b6b",
        "#4ecdc4",
        "#45b7d1",  # Junio
    ]

    # Crear el diagrama de Sankey
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=todos_nodos,
                    color=colores_nodos,
                ),
                link=dict(
                    source=source, target=target, value=value, color="rgba(0,0,0,0.3)"
                ),
            )
        ]
    )

    grupo_nombre = {"G": "Graduados", "A": "Afluentes", "E": "Enrollment"}[
        grupo_seleccionado
    ]
    fig.update_layout(
        title=f"Transiciones de Tipo de Empleo - {grupo_nombre} {periodo} (Marzo → Junio)",
        font_size=12,
        height=500,
    )

    return fig


def calcular_quintiles_hogares(datos_filtrados, periodo, grupo_seleccionado):
    """
    Asigna a cada hogar un quintil según su salario total (Junio 2025).
    Usa hogar_id único a partir de (ced_padre, ced_madre) para evitar duplicados por hermanos.
    """
    if grupo_seleccionado not in ["A", "E"]:
        return {}

    # 1) Rangos de quintiles (ojo a los gaps: ajusté límites para que sean contiguos)
    quintiles_ref = [
        {"Quintil": 1, "Salario_Min": 1.13, "Salario_Max": 642.03},
        {"Quintil": 2, "Salario_Min": 642.03, "Salario_Max": 909.08},
        {"Quintil": 3, "Salario_Min": 909.08, "Salario_Max": 1415.91},
        {"Quintil": 4, "Salario_Min": 1415.91, "Salario_Max": 2491.60},
        {"Quintil": 5, "Salario_Min": 2491.60, "Salario_Max": 20009.99},
    ]

    # 2) Estudiantes del periodo
    df_personas = datos_filtrados["Personas"]
    estudiantes_periodo = df_personas.loc[
        df_personas["periodo"] == periodo, "identificacion"
    ].drop_duplicates()

    # 3) Universo de familiares (solo estudiantes del periodo)
    df_univ = datos_filtrados.get("Universo Familiares", pd.DataFrame())
    if df_univ.empty:
        return {}

    u = df_univ[df_univ["identificacion"].isin(estudiantes_periodo)].copy()
    u["ced_padre"] = (
        u["ced_padre"].astype(str).str.strip().replace({"": "0", "nan": "0"})
    )
    u["ced_madre"] = (
        u["ced_madre"].astype(str).str.strip().replace({"": "0", "nan": "0"})
    )

    # 4) hogar_id único (independiente del orden padre/madre)
    u["hogar_id"] = u.apply(
        lambda r: make_hogar_id(r["ced_padre"], r["ced_madre"]), axis=1
    )
    u = u[u["hogar_id"] != ""].copy()
    if u.empty:
        return {}

    # 5) Mapa hogar_id -> familiares (padre/madre) sin “0”
    pares = []
    for _, r in u.iterrows():
        if r["ced_padre"] != "0":
            pares.append((r["hogar_id"], r["ced_padre"]))
        if r["ced_madre"] != "0":
            pares.append((r["hogar_id"], r["ced_madre"]))
    df_mapa = pd.DataFrame(pares, columns=["hogar_id", "fam_id"]).drop_duplicates()
    if df_mapa.empty:
        return {}

    # 6) Ingresos junio/2025
    df_ing = datos_filtrados.get("Ingresos", pd.DataFrame())
    if df_ing.empty:
        return {}
    df_ing_6 = df_ing[(df_ing["anio"] == 2025) & (df_ing["mes"] == 6)][
        ["identificacion", "salario"]
    ].copy()
    # asegurar numérico
    df_ing_6["salario"] = pd.to_numeric(df_ing_6["salario"], errors="coerce")

    # 7) Salario del hogar = suma (papá + mamá)
    df_merge = df_mapa.merge(
        df_ing_6, left_on="fam_id", right_on="identificacion", how="left"
    )
    df_hogar_sal = df_merge.groupby("hogar_id", as_index=False)["salario"].sum()
    df_hogar_sal = df_hogar_sal.dropna(subset=["salario"])  # type: ignore
    if df_hogar_sal.empty:
        return {}

    # 8) Asignar quintil según rangos
    def asignar_quintil(s):
        for q in quintiles_ref:
            if q["Salario_Min"] <= s <= q["Salario_Max"]:
                return q["Quintil"]
        return None

    df_hogar_sal["quintil"] = df_hogar_sal["salario"].apply(asignar_quintil)

    # 9) Distribución (%)
    counts = df_hogar_sal["quintil"].value_counts().sort_index()
    total = counts.sum()
    return {
        f"Quintil {i}": (counts.get(i, 0) / total * 100 if total else 0)
        for i in range(1, 6)
    }


# Configuración de la página
st.set_page_config(page_title="Empleos e Ingresos", page_icon="💼", layout="wide")

# Título principal
st.title("💼 Empleos e Ingresos")

# Cargar datos
df_vulnerabilidad = cargar_datos_vulnerabilidad()

grupo_seleccionado, facultad_seleccionada, carrera_seleccionada = mostrar_filtros(
    df_vulnerabilidad["Personas"], key_suffix="pagina1"  # Sufijo único para esta página
)

datos_filtrados = aplicar_filtros(
    df_vulnerabilidad, grupo_seleccionado, facultad_seleccionada, carrera_seleccionada
)

# Obtener periodos únicos del grupo seleccionado, ordenados alfabéticamente
df_personas_filtrado = datos_filtrados["Personas"]
periodos = sorted(df_personas_filtrado["periodo"].dropna().unique().tolist())

# Lógica condicional según el grupo seleccionado
if grupo_seleccionado in ["G", "A"]:  # Graduados o Afluentes
    st.subheader("📊 Análisis por Periodos")

    # Si hay al menos 2 periodos, mostrar en columnas
    if len(periodos) >= 2:
        # Dividir en dos columnas para mostrar los periodos
        col1, col_divider, col2 = st.columns([1, 0.05, 1])

        with col1:
            st.write(f"### {periodos[0]}")

            # Si es graduados, calcular empleados activos y desempleados
            if grupo_seleccionado == "G":
                # Calcular métricas de empleo para graduados
                activos, desempleados = calcular_empleo_graduados(
                    datos_filtrados, periodos[0]
                )

                # Mostrar tarjetas de métricas
                col_activos, col_desempleados = st.columns(2)
                with col_activos:
                    tarjeta_simple("Empleados Activos", activos, COLORES["verde"])
                with col_desempleados:
                    tarjeta_simple("Desempleados", desempleados, COLORES["rojo"])

                # Agregar boxplot de salarios
                st.markdown("---")
                st.write("**💰 Distribución de Salarios**")
                fig = crear_boxplot_salarios(
                    datos_filtrados, periodos[0], grupo_seleccionado
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No hay datos de salarios disponibles para este periodo")

                # Agregar tarjetas de quintiles
                st.markdown("---")
                quintiles = calcular_quintiles(
                    datos_filtrados, periodos[0], grupo_seleccionado
                )
                mostrar_tarjetas_quintiles(
                    quintiles, rangos_dict=rangos_quintiles_hogar_dict()
                )

                # Agregar diagrama de Sankey para primera columna de Graduados
                st.markdown("---")
                st.write("**🔄 Transiciones de Tipo de Empleo**")
                fig_sankey = crear_diagrama_sankey(
                    datos_filtrados, periodos[0], grupo_seleccionado
                )
                if fig_sankey:
                    st.plotly_chart(fig_sankey, use_container_width=True)
                else:
                    st.info(
                        "No hay datos suficientes para mostrar transiciones de empleo"
                    )
            else:  # Afluentes
                # Calcular métricas de empleo para familiares
                activos, desempleados, total_hogares, hogares_con_trabajo = (
                    calcular_empleo_familiares(datos_filtrados, periodos[0])
                )

                # Mostrar tarjetas de familiares
                col_activos, col_desempleados = st.columns(2)
                with col_activos:
                    tarjeta_simple(
                        "Empleo Formal - Familiares", activos, COLORES["verde"]
                    )
                with col_desempleados:
                    tarjeta_simple(
                        "Sin Empleo Formal - Familiares", desempleados, COLORES["rojo"]
                    )

                # Separador
                st.markdown("---")

                # Mostrar tarjetas de hogares
                col_total_hogares, col_hogares_trabajo = st.columns(2)
                with col_total_hogares:
                    tarjeta_simple("Total Hogares", total_hogares, COLORES["azul"])
                with col_hogares_trabajo:
                    tarjeta_simple(
                        "Hogares con Trabajo", hogares_con_trabajo, COLORES["amarillo"]
                    )

                # Boxplot por hogar
                st.markdown("---")
                st.write("**🏠 Distribución de Salarios Familiares (por hogar)**")
                fig_hogar = crear_boxplot_salarios_hogares(
                    datos_filtrados, periodos[0], grupo_seleccionado
                )
                if fig_hogar:
                    st.plotly_chart(fig_hogar, use_container_width=True)
                else:
                    st.info(
                        "No hay datos de salarios por hogar disponibles para este periodo"
                    )

                # Agregar tarjetas de quintiles
                st.markdown("---")
                quintiles = calcular_quintiles_hogares(
                    datos_filtrados, periodos[0], grupo_seleccionado
                )
                mostrar_tarjetas_quintiles(
                    quintiles, rangos_dict=rangos_quintiles_hogar_dict()
                )

                # Agregar diagrama de Sankey para primera columna de Afluentes
                st.markdown("---")
                st.write("**🔄 Transiciones de Tipo de Empleo**")
                fig_sankey = crear_diagrama_sankey(
                    datos_filtrados, periodos[0], grupo_seleccionado
                )
                if fig_sankey:
                    st.plotly_chart(fig_sankey, use_container_width=True)
                else:
                    st.info(
                        "No hay datos suficientes para mostrar transiciones de empleo"
                    )

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

        with col2:
            st.write(f"### {periodos[1]}")

            # Si es graduados, calcular empleados activos y desempleados
            if grupo_seleccionado == "G":
                # Calcular métricas de empleo para graduados
                activos, desempleados = calcular_empleo_graduados(
                    datos_filtrados, periodos[1]
                )

                # Mostrar tarjetas de métricas
                col_activos, col_desempleados = st.columns(2)
                with col_activos:
                    tarjeta_simple("Empleados Activos", activos, COLORES["verde"])
                with col_desempleados:
                    tarjeta_simple("Desempleados", desempleados, COLORES["rojo"])

                # Agregar boxplot de salarios
                st.markdown("---")
                st.write("**💰 Distribución de Salarios**")
                fig = crear_boxplot_salarios(
                    datos_filtrados, periodos[1], grupo_seleccionado
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No hay datos de salarios disponibles para este periodo")

                # Agregar tarjetas de quintiles
                st.markdown("---")
                quintiles = calcular_quintiles(
                    datos_filtrados, periodos[1], grupo_seleccionado
                )
                mostrar_tarjetas_quintiles(
                    quintiles, rangos_dict=rangos_quintiles_hogar_dict()
                )

                # Agregar diagrama de Sankey para segunda columna de Graduados
                st.markdown("---")
                st.write("**🔄 Transiciones de Tipo de Empleo**")
                fig_sankey = crear_diagrama_sankey(
                    datos_filtrados, periodos[1], grupo_seleccionado
                )
                if fig_sankey:
                    st.plotly_chart(fig_sankey, use_container_width=True)
                else:
                    st.info(
                        "No hay datos suficientes para mostrar transiciones de empleo"
                    )
            else:  # Afluentes
                # Calcular métricas de empleo para familiares
                activos, desempleados, total_hogares, hogares_con_trabajo = (
                    calcular_empleo_familiares(datos_filtrados, periodos[1])
                )

                # Mostrar tarjetas de familiares
                col_activos, col_desempleados = st.columns(2)
                with col_activos:
                    tarjeta_simple(
                        "Empleo Formal - Familiares", activos, COLORES["verde"]
                    )
                with col_desempleados:
                    tarjeta_simple(
                        "Sin Empleo Formal - Familiares", desempleados, COLORES["rojo"]
                    )

                # Separador
                st.markdown("---")

                # Mostrar tarjetas de hogares
                col_total_hogares, col_hogares_trabajo = st.columns(2)
                with col_total_hogares:
                    tarjeta_simple("Total Hogares", total_hogares, COLORES["azul"])
                with col_hogares_trabajo:
                    tarjeta_simple(
                        "Hogares con Trabajo", hogares_con_trabajo, COLORES["amarillo"]
                    )

                # Boxplot por hogar
                st.markdown("---")
                st.write("**🏠 Distribución de Salarios Familiares (por hogar)**")
                fig_hogar = crear_boxplot_salarios_hogares(
                    datos_filtrados, periodos[1], grupo_seleccionado
                )
                if fig_hogar:
                    st.plotly_chart(fig_hogar, use_container_width=True)
                else:
                    st.info(
                        "No hay datos de salarios por hogar disponibles para este periodo"
                    )

                # Agregar tarjetas de quintiles
                st.markdown("---")
                quintiles = calcular_quintiles_hogares(
                    datos_filtrados, periodos[1], grupo_seleccionado
                )
                mostrar_tarjetas_quintiles(
                    quintiles, rangos_dict=rangos_quintiles_hogar_dict()
                )

                # Agregar diagrama de Sankey para segunda columna de Afluentes
                st.markdown("---")
                st.write("**🔄 Transiciones de Tipo de Empleo**")
                fig_sankey = crear_diagrama_sankey(
                    datos_filtrados, periodos[1], grupo_seleccionado
                )
                if fig_sankey:
                    st.plotly_chart(fig_sankey, use_container_width=True)
                else:
                    st.info(
                        "No hay datos suficientes para mostrar transiciones de empleo"
                    )
    else:
        # Si hay un solo periodo o ninguno
        if periodos:
            st.write(f"### {periodos[0]}")
            st.write(f"Contenido del {periodos[0]}")
        else:
            st.write("No hay periodos disponibles para este grupo")

else:  # Enrollment (E)
    st.subheader("📊 Análisis General - Enrollment")

    if periodos:
        periodo_sel = periodos[0]
        st.write(f"### {periodo_sel}")

        # === NUEVOS FILTROS ===
        c1, c2, c3 = st.columns(3)

        with c1:
            cant_papas_opt = st.selectbox(
                "Cantidad de papás en el hogar",
                options=["Todos", 1, 2],
                index=0,
                help="Hogares con 1 o 2 representantes (excluye huérfanos).",
            )
            # Normaliza: None = no filtrar
            cant_papas = None if cant_papas_opt == "Todos" else int(cant_papas_opt)

        with c2:
            cant_papas_trab_opt = st.selectbox(
                "Cantidad de papás trabajando (JUN/2025)",
                options=["Todos", 0, 1, 2],
                index=0,
                help="Se considera 'trabajando' si aparece en Ingresos 2025-06 con Relación de Dependencia o Afiliación Voluntaria.",
            )
            # Normaliza: None = no filtrar
            cant_papas_trab = (
                None if cant_papas_trab_opt == "Todos" else int(cant_papas_trab_opt)
            )

        with c3:
            tipo_empleo_sel = st.selectbox(
                "Tipo de empleo (JUN/2025)",
                options=["Todos"] + EMPLEOS_TODOS,
                index=0,
                help="‘Desconocido’ = no aparece en Ingresos del mes 6/2025.",
            )
            # La función espera lista[str]
            tipos_empleo_sel = (
                EMPLEOS_TODOS if tipo_empleo_sel == "Todos" else [tipo_empleo_sel]
            )

        # Construir universo filtrado según los 3 filtros
        universo = construir_enrollment_filtrado(
            datos_filtrados, periodo_sel, cant_papas, cant_papas_trab, tipos_empleo_sel
        )

        fam_ids = universo["familiares_ids"]
        hogares_ids = universo["hogares_ids"]
        df_hogar_sal = universo["df_hogar_salarios"]
        df_emp = universo.get("df_emp", pd.DataFrame())

        # ============ TARJETAS PRINCIPALES ============
        total_familiares = universo["resumen"]["total_familiares"]
        activos = int(df_emp["trabaja_mes6"].sum()) if not df_emp.empty else 0
        desempleados = max(total_familiares - activos, 0)

        col_activos, col_desempleados = st.columns(2)
        with col_activos:
            tarjeta_simple("Empleo Formal - Familiares", activos, COLORES["verde"])
        with col_desempleados:
            tarjeta_simple(
                "Sin Empleo Formal - Familiares", desempleados, COLORES["rojo"]
            )

        # Separador
        st.markdown("---")

        # ============ TARJETAS DE HOGARES ============
        total_hogares = universo["resumen"]["total_hogares"]
        hogares_con_trabajo = 0
        if not df_emp.empty:
            hogares_con_trabajo = (
                df_emp.groupby("hogar_id")["trabaja_mes6"].max().sum()
            )  # ≥1 trabajando

        col_total_hogares, col_hogares_trabajo = st.columns(2)
        with col_total_hogares:
            tarjeta_simple("Total Hogares (filtrados)", total_hogares, COLORES["azul"])
        with col_hogares_trabajo:
            tarjeta_simple(
                "Hogares con Trabajo", int(hogares_con_trabajo), COLORES["amarillo"]
            )

        # ============ BOXPLOT POR HOGAR ============
        st.markdown("---")
        st.write("**🏠 Distribución de Salarios Familiares (por hogar)**")
        if not df_hogar_sal.empty:
            salarios = filtrar_outliers_iqr(df_hogar_sal["salario"], k=1.5)
            if not salarios.empty:
                fig_hogar = go.Figure()
                fig_hogar.add_trace(
                    go.Box(
                        y=salarios,
                        name="Salarios por Hogar",
                        boxpoints=False,
                        marker_color="lightgreen",
                        line_color="darkgreen",
                    )
                )
                fig_hogar.update_layout(
                    title=f"Distribución de Salarios por Hogar - Familiares Enrollment {periodo_sel}",
                    yaxis_title="Salario (USD)",
                    height=400,
                    showlegend=False,
                    yaxis=dict(tickformat="$,.0f"),
                )
                st.plotly_chart(fig_hogar, use_container_width=True)
            else:
                st.info("No hay salarios válidos tras filtrar outliers.")
        else:
            st.info("No hay datos de salarios por hogar con los filtros actuales.")

        # ============ QUINTILES POR HOGAR ============
        st.markdown("---")
        if not df_hogar_sal.empty:

            def asignar_quintil(s):
                for q in RANGOS_QUINTILES_HOGAR:
                    if q["Salario_Min"] <= s <= q["Salario_Max"]:
                        return q["Quintil"]
                return None

            df_q = df_hogar_sal.copy()
            df_q["quintil"] = df_q["salario"].apply(asignar_quintil)
            counts = df_q["quintil"].value_counts().sort_index()
            total = counts.sum()
            quintiles = {
                f"Quintil {i}": (counts.get(i, 0) / total * 100 if total else 0)
                for i in range(1, 6)
            }
            mostrar_tarjetas_quintiles(
                quintiles, rangos_dict=rangos_quintiles_hogar_dict()
            )
        else:
            st.info("No hay hogares para calcular quintiles con los filtros actuales.")

        # ============ SANKEY (Mar→Jun) SOLO PERSONAS FILTRADAS ============
        st.markdown("---")
        st.write("**🔄 Transiciones de Tipo de Empleo (Marzo → Junio)**")
        df_ing = datos_filtrados.get("Ingresos", pd.DataFrame())
        if not df_ing.empty and fam_ids:
            df_ing_fam = df_ing[df_ing["identificacion"].isin(fam_ids)]
            df_mes3 = df_ing_fam[
                (df_ing_fam["anio"] == 2025) & (df_ing_fam["mes"] == 3)
            ]
            df_mes6 = df_ing_fam[
                (df_ing_fam["anio"] == 2025) & (df_ing_fam["mes"] == 6)
            ]

            empleo_mes3 = dict(zip(df_mes3["identificacion"], df_mes3["tipo_empleo"]))
            empleo_mes6 = dict(zip(df_mes6["identificacion"], df_mes6["tipo_empleo"]))

            tipos = ["Relacion de Dependencia", "Afiliacion Voluntaria", "Desconocido"]

            nodos_marzo = [f"{t} (Marzo)" for t in tipos]
            nodos_junio = [f"{t} (Junio)" for t in tipos]
            todos_nodos = nodos_marzo + nodos_junio
            idx = {n: i for i, n in enumerate(todos_nodos)}

            from collections import Counter

            trans = []
            for p in fam_ids:
                o = empleo_mes3.get(p, "Desconocido")
                d = empleo_mes6.get(p, "Desconocido")
                if o not in tipos:
                    o = "Desconocido"
                if d not in tipos:
                    d = "Desconocido"
                trans.append((o, d))
            c = Counter(trans)

            src, tgt, val = [], [], []
            for (o, d), k in c.items():
                if k > 0:
                    src.append(idx[f"{o} (Marzo)"])
                    tgt.append(idx[f"{d} (Junio)"])
                    val.append(k)

            if val:
                fig_sankey = go.Figure(
                    data=[
                        go.Sankey(
                            node=dict(
                                pad=15,
                                thickness=20,
                                line=dict(color="black", width=0.5),
                                label=todos_nodos,
                                color=["#ff6b6b", "#4ecdc4", "#45b7d1"] * 2,
                            ),
                            link=dict(
                                source=src,
                                target=tgt,
                                value=val,
                                color="rgba(0,0,0,0.3)",
                            ),
                        )
                    ]
                )
                fig_sankey.update_layout(
                    title=f"Transiciones de Tipo de Empleo - Familiares Enrollment {periodo_sel} (Marzo → Junio)",
                    font_size=12,
                    height=500,
                )
                st.plotly_chart(fig_sankey, use_container_width=True)
            else:
                st.info("No hay transiciones para mostrar con los filtros actuales.")
        else:
            st.info("No hay datos suficientes para mostrar transiciones de empleo.")
    else:
        st.write("No hay periodos disponibles para este grupo")
