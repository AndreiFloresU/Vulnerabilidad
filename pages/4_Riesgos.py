import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.carga_datos import cargar_datos_vulnerabilidad
from utils.filtros import (
    aplicar_filtros,
    obtener_facultades_por_grupo,
    obtener_carreras_por_grupo_y_facultad,
)
from utils.tarjetas import tarjeta_simple, COLORES
import numpy as np
from utils.hogarUnico import make_hogar_id  # 👈 para identificar hogares únicos
from typing import Tuple, List
import re

# ====== Constantes de empleo (para los filtros) ======
EMPLEOS_VALIDOS = ["Relacion de Dependencia", "Afiliacion Voluntaria"]
EMPLEOS_TODOS = EMPLEOS_VALIDOS + ["Desconocido"]


def _parse_rgba_str(rgba_str: str) -> Tuple[int, int, int, float]:
    m = re.match(r"rgba\((\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)\)", rgba_str.strip())
    if not m:
        raise ValueError(f"RGBA inválido: {rgba_str}")
    r, g, b, a = m.groups()
    return int(r), int(g), int(b), float(a)


def _mix_with_white(rgb: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    # t=0 => color base, t=1 => blanco
    r0, g0, b0 = rgb
    r = int(round(r0 * (1.0 - t) + 255 * t))
    g = int(round(g0 * (1.0 - t) + 255 * t))
    b = int(round(b0 * (1.0 - t) + 255 * t))
    return (r, g, b)


def generar_paleta_pastel(desde_rgba: str, n: int = 10) -> List[str]:
    base_r, base_g, base_b, _ = _parse_rgba_str(desde_rgba)
    base_rgb: Tuple[int, int, int] = (base_r, base_g, base_b)
    ts = np.linspace(0.0, 0.82, n)  # 0 = más fuerte, 0.82 = muy pastel
    colores: List[str] = []
    for t in ts:
        r, g, b = _mix_with_white(base_rgb, float(t))
        colores.append(f"rgba({r},{g},{b},1.0)")
    return colores


PALETA_AZUL_PASTEL_5 = generar_paleta_pastel("rgba(0,112,192,1.0)", n=5)


def calcular_vulnerabilidad_estudiantes(datos_filtrados, periodo):
    df_personas = datos_filtrados["Personas"]
    estudiantes_periodo = df_personas[df_personas["periodo"] == periodo].copy()
    if estudiantes_periodo.empty:
        return pd.DataFrame()

    # Flags
    estudiantes_periodo["vulnerable"] = False
    estudiantes_periodo["en_riesgo"] = False
    estudiantes_periodo["motivos_vulnerabilidad"] = ""
    estudiantes_periodo["contador_riesgos"] = 0

    # Datos
    df_universo = datos_filtrados.get("Universo Familiares", pd.DataFrame())
    df_ingresos = datos_filtrados.get("Ingresos", pd.DataFrame())
    df_deudas = datos_filtrados.get("Deudas", pd.DataFrame())

    # Precalcular estructuras auxiliares (pueden quedar vacías)
    personas_con_ingresos = set()
    if not df_ingresos.empty:
        ingresos_mes6 = df_ingresos[
            (df_ingresos["anio"] == 2025) & (df_ingresos["mes"] == 6)
        ]
        if not ingresos_mes6.empty and "salario" in ingresos_mes6.columns:
            personas_con_ingresos = set(
                ingresos_mes6[ingresos_mes6["salario"] > 0]["identificacion"]
            )

    # Precalcular deudas (julio 2025)
    deudas_mes7 = pd.DataFrame()
    if not df_deudas.empty:
        deudas_mes7 = df_deudas[(df_deudas["anio"] == 2025) & (df_deudas["mes"] == 7)]

    # Merge (si no hay universo, igual iteramos marcando "sin info familiar")
    estudiantes_con_familia = (
        estudiantes_periodo.merge(df_universo, on="identificacion", how="left")
        if not df_universo.empty
        else estudiantes_periodo.copy()
    )

    def tiene_cedula_valida(ced):
        return pd.notna(ced) and str(ced) != "0"

    for _, est in estudiantes_con_familia.iterrows():
        motivos = []
        contador = 0

        # Cedulas normalizadas
        ced_padre = est.get("ced_padre")
        ced_madre = est.get("ced_madre")
        tiene_padre = tiene_cedula_valida(ced_padre)
        tiene_madre = tiene_cedula_valida(ced_madre)

        # Índice original para escribir
        idx_original = estudiantes_periodo.index[
            estudiantes_periodo["identificacion"] == est["identificacion"]
        ][0]

        # Regla solicitada: sin familiares => Alta vulnerabilidad
        if not tiene_padre and not tiene_madre:
            estudiantes_periodo.loc[idx_original, ["vulnerable", "en_riesgo"]] = [
                True,
                False,
            ]
            estudiantes_periodo.loc[idx_original, "contador_riesgos"] = 2
            estudiantes_periodo.loc[idx_original, "motivos_vulnerabilidad"] = (
                "Sin información familiar"
            )
            continue

        # --- Criterio 1: Familia sin empleo (junio 2025) ---
        # Nota: solo se evalúa si existen padres y tenemos el set de ingresos
        if personas_con_ingresos is not None:
            padre_sin_empleo = tiene_padre and (ced_padre not in personas_con_ingresos)
            madre_sin_empleo = tiene_madre and (ced_madre not in personas_con_ingresos)

            if tiene_padre and tiene_madre:
                if padre_sin_empleo and madre_sin_empleo:
                    motivos.append("Familia sin empleo")
                    contador += 1
            elif tiene_padre and not tiene_madre:
                if padre_sin_empleo:
                    motivos.append("Familia sin empleo")
                    contador += 1
            elif not tiene_padre and tiene_madre:
                if madre_sin_empleo:
                    motivos.append("Familia sin empleo")
                    contador += 1

        # --- Criterio 2: Deuda familiar crítica (D/E) en julio 2025 ---
        if not deudas_mes7.empty and "cod_calificacion" in deudas_mes7.columns:
            cedulas_familia = []
            if tiene_padre:
                cedulas_familia.append(ced_padre)
            if tiene_madre:
                cedulas_familia.append(ced_madre)

            if cedulas_familia:
                deudas_fam = deudas_mes7[
                    deudas_mes7["identificacion"].isin(cedulas_familia)
                ]

                # Verifica que haya deuda D/E
                if (
                    not deudas_fam.empty
                    and deudas_fam["cod_calificacion"].isin(["D", "E"]).any()
                ):
                    # Calcular deuda total del hogar
                    deuda_total = deudas_fam["valor"].sum()

                    # Calcular ingreso anual del hogar (junio 2025, multiplicado por 14)
                    ingreso_anual = 0
                    if not df_ingresos.empty and "salario" in df_ingresos.columns:
                        ingresos_junio = df_ingresos[
                            (df_ingresos["anio"] == 2025) & (df_ingresos["mes"] == 6)
                        ]
                        if not ingresos_junio.empty:
                            ingresos_fam = ingresos_junio[
                                ingresos_junio["identificacion"].isin(cedulas_familia)
                            ]
                            ingreso_anual = ingresos_fam["salario"].sum() * 14

                    # Aplica condición del ratio deuda/ingreso
                    if ingreso_anual > 0 and (deuda_total / ingreso_anual) >= 2.90:
                        motivos.append("Deuda familiar crítica (D/E)")
                        contador += 1

        # --- Criterio 3: Bajó de quintil (marzo → junio 2025) ---
        if not df_ingresos.empty and "quintil" in df_ingresos.columns:
            ingresos_marzo = df_ingresos[
                (df_ingresos["anio"] == 2025) & (df_ingresos["mes"] == 3)
            ]
            ingresos_junio = df_ingresos[
                (df_ingresos["anio"] == 2025) & (df_ingresos["mes"] == 6)
            ]

            if not ingresos_marzo.empty and not ingresos_junio.empty:
                quintiles_marzo = []
                quintiles_junio = []

                for ced in [ced_padre, ced_madre]:
                    if tiene_cedula_valida(ced):
                        # marzo
                        if ced in ingresos_marzo["identificacion"].values:
                            q_mar = ingresos_marzo.loc[
                                ingresos_marzo["identificacion"] == ced, "quintil"
                            ].values[0]
                        else:
                            q_mar = 0  # no tiene ingresos => 0
                        quintiles_marzo.append(q_mar)

                        # junio
                        if ced in ingresos_junio["identificacion"].values:
                            q_jun = ingresos_junio.loc[
                                ingresos_junio["identificacion"] == ced, "quintil"
                            ].values[0]
                        else:
                            q_jun = 0
                        quintiles_junio.append(q_jun)

                # Caso con dos familiares: promedio
                if len(quintiles_marzo) == 2:
                    q_mar = sum(quintiles_marzo) / 2
                    q_jun = sum(quintiles_junio) / 2
                # Caso con un solo familiar existente
                elif len(quintiles_marzo) == 1:
                    q_mar = quintiles_marzo[0]
                    q_jun = quintiles_junio[0]
                else:
                    q_mar, q_jun = None, None

                if q_mar is not None and q_jun is not None and q_jun < q_mar:
                    motivos.append("Familia bajó de quintil (marzo-junio)")
                    contador += 1

        # --- Escritura de resultado para el estudiante ---
        if contador > 0:
            if contador >= 2:
                estudiantes_periodo.loc[idx_original, ["vulnerable", "en_riesgo"]] = [
                    True,
                    False,
                ]
            else:
                estudiantes_periodo.loc[idx_original, ["vulnerable", "en_riesgo"]] = [
                    False,
                    True,
                ]
            estudiantes_periodo.loc[idx_original, "contador_riesgos"] = contador
            estudiantes_periodo.loc[idx_original, "motivos_vulnerabilidad"] = "; ".join(
                motivos
            )

    return estudiantes_periodo


def crear_barras_facultades_vulnerables(estudiantes_vulnerables, periodo):
    """
    Crea gráfico de barras con top 5 facultades con más estudiantes vulnerables
    """
    if estudiantes_vulnerables.empty:
        return None

    # Calcular estadísticas por facultad
    stats_facultad = (
        estudiantes_vulnerables.groupby("facultad")
        .agg(
            {
                "identificacion": "count",  # Total estudiantes
                "vulnerable": "sum",  # Estudiantes vulnerables
                "en_riesgo": "sum",  # Estudiantes en riesgo
            }
        )
        .reset_index()
    )

    stats_facultad.columns = [
        "facultad",
        "total_estudiantes",
        "vulnerables",
        "en_riesgo",
    ]

    # Agregar total de estudiantes con algún tipo de riesgo
    stats_facultad["total_con_riesgo"] = (
        stats_facultad["vulnerables"] + stats_facultad["en_riesgo"]
    )

    # Filtrar solo facultades con estudiantes en algún tipo de riesgo
    stats_facultad = stats_facultad[stats_facultad["total_con_riesgo"] > 0]

    if stats_facultad.empty:
        return None

    # Ordenar por total con riesgo y tomar top 5
    top_facultades = stats_facultad.sort_values(
        "total_con_riesgo", ascending=False
    ).head(5)

    # Cantidad real de barras (por si hay <5)
    k = len(top_facultades)

    # Crear gráfico de barras
    fig = px.bar(
        top_facultades,
        x="total_con_riesgo",
        y="facultad",
        orientation="h",
        title=f"Top 5 Facultades con Más Estudiantes en Situación de Riesgo - Enrollment {periodo}",
        labels={
            "total_con_riesgo": "Número de Estudiantes en Riesgo",
            "facultad": "Facultad",
        },
        text="total_con_riesgo",
        color="facultad",
        color_discrete_sequence=PALETA_AZUL_PASTEL_5[:k],
    )

    # Personalizar
    fig.update_traces(
        textposition="inside",
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b><br>"
        + "Total en riesgo: %{x}<br>"
        + "Alta vulnerabilidad: %{marker.color}<br>"
        + "<extra></extra>",
        marker=dict(line=dict(color="white", width=1)),
    )

    fig.update_layout(
        height=400,
        yaxis=dict(categoryorder="total ascending"),
        xaxis=dict(title="Número de Estudiantes en Riesgo"),
        coloraxis_colorbar=dict(title="Estudiantes con Alta Vulnerabilidad"),
    )

    return fig


# ====== Helper: aplicar los 3 filtros al Universo (solo para Enrollment) ======
def _filtrar_universo_enrollment(
    datos_filtrados: dict,
    periodo: str,
    cant_papas: int | None,
    cant_papas_trab: int | None,
    tipo_empleo_sel: str | None,
) -> pd.DataFrame:
    """
    Devuelve el 'Universo Familiares' filtrado por:
      - cant_papas: 0/1/2 (None = Todos)
      - cant_papas_trab: 0/1/2 (None = Todos)
      - tipo_empleo_sel: "Todos" o uno de EMPLEOS_TODOS
    NO toca la lógica de vulnerabilidad; solo reduce el universo antes del merge.
    """
    df_u = datos_filtrados.get("Universo Familiares", pd.DataFrame())
    if df_u.empty:
        return df_u

    # Estudiantes del periodo
    df_personas = datos_filtrados["Personas"]
    ids = df_personas.loc[
        df_personas["periodo"] == periodo, "identificacion"
    ].drop_duplicates()
    u = df_u[df_u["identificacion"].isin(ids)].copy()

    # Normalizar
    u["ced_padre"] = (
        u["ced_padre"].astype(str).str.strip().replace({"": "0", "nan": "0"})
    )
    u["ced_madre"] = (
        u["ced_madre"].astype(str).str.strip().replace({"": "0", "nan": "0"})
    )

    # Cantidad de papás en el hogar
    u["n_papas"] = (u["ced_padre"].ne("0")).astype(int) + (
        u["ced_madre"].ne("0")
    ).astype(int)
    if cant_papas in (0, 1, 2):
        u = u[u["n_papas"] == cant_papas]
        if u.empty:
            return u

    # Si NO hay filtros de empleo, devolvemos tal cual (tras n_papas)
    tiene_filtro_empleo = (cant_papas_trab in (0, 1, 2)) or (
        tipo_empleo_sel is not None and tipo_empleo_sel != "Todos"
    )
    if not tiene_filtro_empleo:
        return u

    # Construir hogar_id para filtrar por empleo
    u["hogar_id"] = u.apply(
        lambda r: make_hogar_id(r["ced_padre"], r["ced_madre"]), axis=1
    )

    # Mapa hogar -> familiares (solo IDs válidos)
    pares = []
    for _, r in u.iterrows():
        if r["ced_padre"] != "0":
            pares.append((r["hogar_id"], r["ced_padre"]))
        if r["ced_madre"] != "0":
            pares.append((r["hogar_id"], r["ced_madre"]))
    df_mapa = pd.DataFrame(pares, columns=["hogar_id", "fam_id"]).drop_duplicates()

    if df_mapa.empty:
        # Si hay filtro de empleo y no hay fams, no pasa nadie
        return u.iloc[0:0]

    # Ingresos JUN/2025 (para tipo de empleo y "trabajando")
    df_ing = datos_filtrados.get("Ingresos", pd.DataFrame())
    if df_ing.empty:
        return u.iloc[0:0]
    ing6 = df_ing[(df_ing["anio"] == 2025) & (df_ing["mes"] == 6)].copy()
    ing6["tipo_empleo"] = ing6["tipo_empleo"].astype(str).str.strip()

    df_emp = df_mapa.merge(
        ing6[["identificacion", "tipo_empleo"]],
        left_on="fam_id",
        right_on="identificacion",
        how="left",
    )
    df_emp["tipo_empleo_mes6"] = df_emp["tipo_empleo"].where(
        df_emp["tipo_empleo"].isin(EMPLEOS_VALIDOS), "Desconocido"
    )
    df_emp["trabaja_mes6"] = df_emp["tipo_empleo_mes6"].isin(EMPLEOS_VALIDOS)

    # Filtro por tipo de empleo (si aplica)
    if tipo_empleo_sel is not None and tipo_empleo_sel != "Todos":
        df_emp = df_emp[df_emp["tipo_empleo_mes6"] == tipo_empleo_sel]
        if df_emp.empty:
            return u.iloc[0:0]

    # Cantidad de papás trabajando por hogar (si aplica)
    agg = df_emp.groupby("hogar_id", as_index=False).agg(n_trab=("trabaja_mes6", "sum"))

    if cant_papas_trab in (0, 1, 2):
        agg = agg[agg["n_trab"] == cant_papas_trab]
        if agg.empty:
            return u.iloc[0:0]

    hogares_ok = set(agg["hogar_id"])
    u_filtrado = u[u["hogar_id"].isin(hogares_ok)].copy()
    return u_filtrado


# =========================
# Página
# =========================
st.set_page_config(page_title="Análisis de Riesgos", page_icon="⚠️", layout="wide")

# Título principal
st.title("⚠️ Análisis de Riesgos")

# Cargar datos
df_vulnerabilidad = cargar_datos_vulnerabilidad()

# --- Filtros personalizados (solo Enrollment) ---
st.header("🔍 Filtros")
col1, col2, col3 = st.columns(3)

with col1:
    grupo_seleccionado = st.selectbox(
        "Grupo de interés:",
        options=["E"],  # solo Enrollment
        format_func=lambda x: "Enrollment",
        index=0,
        disabled=True,
        key="grupo_interes_riesgos",
    )

with col2:
    facultades_filtradas = obtener_facultades_por_grupo(
        df_vulnerabilidad["Personas"], grupo_seleccionado
    )
    facultad_seleccionada = st.selectbox(
        "Facultad:",
        options=facultades_filtradas,
        index=0,
        key="facultad_seleccionada_riesgos",
    )

with col3:
    carreras_filtradas = obtener_carreras_por_grupo_y_facultad(
        df_vulnerabilidad["Personas"], grupo_seleccionado, facultad_seleccionada
    )
    carrera_seleccionada = st.selectbox(
        "Carrera:",
        options=carreras_filtradas,
        index=0,
        key="carrera_seleccionada_riesgos",
    )

# --- Aplicar filtros base (grupo/facultad/carrera) ---
datos_filtrados = aplicar_filtros(
    df_vulnerabilidad, grupo_seleccionado, facultad_seleccionada, carrera_seleccionada
)

# --- Periodos disponibles ---
df_personas_filtrado = datos_filtrados["Personas"]
periodos = sorted(df_personas_filtrado["periodo"].dropna().unique().tolist())

if periodos:
    st.subheader("📊 Análisis de Vulnerabilidad - Enrollment")

    periodo = periodos[0]  # primer periodo disponible
    st.write(f"### {periodo}")

    # ====== NUEVOS 3 FILTROS (solo Enrollment) ======
    c1, c2, c3 = st.columns(3)

    with c1:
        cant_papas_opt = st.selectbox(
            "Cantidad de papás en el hogar",
            options=["Todos", 0, 1, 2],
            index=0,
            help="Número de representantes (0 = sin padres registrados).",
            key="cant_papas_riesgos",
        )
        cant_papas = None if cant_papas_opt == "Todos" else int(cant_papas_opt)

    with c2:
        cant_papas_trab_opt = st.selectbox(
            "Cantidad de papás trabajando (JUN/2025)",
            options=["Todos", 0, 1, 2],
            index=0,
            help="Se considera 'trabajando' si aparece en Ingresos 2025-06 con Relación de Dependencia o Afiliación Voluntaria.",
            key="cant_papas_trab_riesgos",
        )
        cant_papas_trab = (
            None if cant_papas_trab_opt == "Todos" else int(cant_papas_trab_opt)
        )

    with c3:
        tipo_empleo_sel = st.selectbox(
            "Tipo de empleo (JUN/2025)",
            options=["Todos"] + EMPLEOS_TODOS,
            index=0,
            help="‘Desconocido’ = no aparece en Ingresos del mes 6/2025.",
            key="tipo_empleo_riesgos",
        )

    # Filtrar SOLO el Universo por los 3 filtros (sin tocar la lógica de vulnerabilidad)
    df_universo_filtrado = _filtrar_universo_enrollment(
        datos_filtrados, periodo, cant_papas, cant_papas_trab, tipo_empleo_sel
    )

    # Copia superficial para no mutar 'datos_filtrados' original
    datos_filtrados_f = dict(datos_filtrados)
    datos_filtrados_f["Universo Familiares"] = df_universo_filtrado

    # 🔧 NUEVO: si hay algún filtro activo (0/1/2 papás, 0/1/2 papás trabajando, o tipo de empleo específico),
    # entonces limita también los estudiantes del PERIODO ACTIVO a quienes están en el universo filtrado.
    aplican_filtros = (
        (cant_papas in (0, 1, 2))
        or (cant_papas_trab in (0, 1, 2))
        or (tipo_empleo_sel is not None and tipo_empleo_sel != "Todos")
    )

    if aplican_filtros:
        ids_ok = set(df_universo_filtrado["identificacion"].unique())
        dfp = datos_filtrados_f["Personas"].copy()

        # Mantener intactos otros periodos. En el periodo activo, conservar solo ids_ok
        mask = (dfp["periodo"] != periodo) | (dfp["identificacion"].isin(ids_ok))
        datos_filtrados_f["Personas"] = dfp[mask]

    # Calcular vulnerabilidad (lógica original intacta)
    estudiantes_vulnerables = calcular_vulnerabilidad_estudiantes(
        datos_filtrados_f, periodo
    )

    if not estudiantes_vulnerables.empty:
        # Métricas
        total_estudiantes = len(estudiantes_vulnerables)
        estudiantes_alta_vulnerabilidad = estudiantes_vulnerables["vulnerable"].sum()
        estudiantes_en_situacion_riesgo = estudiantes_vulnerables["en_riesgo"].sum()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            tarjeta_simple("Total Estudiantes", f"{total_estudiantes}", COLORES["azul"])
        with col2:
            tarjeta_simple(
                "Alta Vulnerabilidad",
                f"{estudiantes_alta_vulnerabilidad}",
                COLORES["rojo"],
            )
        with col3:
            tarjeta_simple(
                "En Situación de Riesgo",
                f"{estudiantes_en_situacion_riesgo}",
                COLORES["naranja"],
            )
        with col4:
            estudiantes_seguros = (
                total_estudiantes
                - estudiantes_alta_vulnerabilidad
                - estudiantes_en_situacion_riesgo
            )
            tarjeta_simple(
                "Sin Riesgo Identificado", f"{estudiantes_seguros}", COLORES["verde"]
            )

        st.markdown("---")

        # Barras por facultad
        fig_barras = crear_barras_facultades_vulnerables(
            estudiantes_vulnerables, periodo
        )
        if fig_barras:
            st.plotly_chart(fig_barras, use_container_width=True)
        else:
            st.info(
                "No se encontraron estudiantes en situación vulnerable para mostrar el análisis por facultades"
            )

        # Detalle de estudiantes
        if estudiantes_alta_vulnerabilidad > 0 or estudiantes_en_situacion_riesgo > 0:
            st.markdown("---")
            st.subheader("📋 Detalle de Estudiantes Vulnerables")

            # Alta vulnerabilidad
            estudiantes_riesgo_alto = estudiantes_vulnerables[
                estudiantes_vulnerables["vulnerable"] == True
            ][
                [
                    "identificacion",
                    "facultad",
                    "carrera_homologada",
                    "motivos_vulnerabilidad",
                ]
            ]

            if not estudiantes_riesgo_alto.empty:
                st.write(
                    "#### 🔴 Estudiantes con Alta Vulnerabilidad (2 o más condiciones)"
                )
                estudiantes_riesgo_alto.columns = [
                    "Identificación",
                    "Facultad",
                    "Carrera",
                    "Motivos de Vulnerabilidad",
                ]
                st.dataframe(estudiantes_riesgo_alto, use_container_width=True)

            # Situación de riesgo
            estudiantes_riesgo_medio = estudiantes_vulnerables[
                estudiantes_vulnerables["en_riesgo"] == True
            ][
                [
                    "identificacion",
                    "facultad",
                    "carrera_homologada",
                    "motivos_vulnerabilidad",
                ]
            ]

            if not estudiantes_riesgo_medio.empty:
                st.write("#### 🟡 Estudiantes en Situación de Riesgo (1 condición)")
                estudiantes_riesgo_medio.columns = [
                    "Identificación",
                    "Facultad",
                    "Carrera",
                    "Motivos de Vulnerabilidad",
                ]
                st.dataframe(estudiantes_riesgo_medio, use_container_width=True)

    else:
        st.info("No hay datos de estudiantes disponibles para este periodo")
else:
    st.write("No hay periodos disponibles para Enrollment")
