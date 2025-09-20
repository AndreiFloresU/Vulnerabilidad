import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.carga_datos import cargar_datos_vulnerabilidad
from utils.filtros import (
    inicializar_filtros,
    aplicar_filtros,
    mostrar_filtros_en_pagina,
)
from utils.tarjetas import tarjeta_simple, COLORES


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
    Calcula empleados activos y desempleados para familiares (afluentes/enrollment) en un periodo específico

    Args:
        datos_filtrados: Diccionario con los datos filtrados
        periodo: Periodo específico a analizar

    Returns:
        tuple: (activos, desempleados, total_hogares, hogares_con_trabajo)
    """
    # Obtener estudiantes del periodo específico
    df_personas = datos_filtrados["Personas"]
    estudiantes_periodo = df_personas[df_personas["periodo"] == periodo][
        "identificacion"
    ].drop_duplicates()

    # Obtener universo de familiares
    df_universo = datos_filtrados.get("Universo Familiares", pd.DataFrame())

    if df_universo.empty:
        return 0, 0, 0, 0

    # Filtrar universo por estudiantes del periodo
    universo_periodo = df_universo[
        df_universo["identificacion"].isin(estudiantes_periodo)
    ]

    # Calcular Total Hogares (estudiantes que tienen al menos un pariente)
    hogares_con_familiares = universo_periodo[
        (universo_periodo["ced_padre"] != "0") | (universo_periodo["ced_madre"] != "0")
    ]
    total_hogares = len(hogares_con_familiares["identificacion"].drop_duplicates())

    # Obtener cédulas únicas de padres y madres (excluyendo los 0)
    cedulas_padres = universo_periodo["ced_padre"][
        universo_periodo["ced_padre"] != "0"
    ].drop_duplicates()
    cedulas_madres = universo_periodo["ced_madre"][
        universo_periodo["ced_madre"] != "0"
    ].drop_duplicates()

    # Combinar cédulas de padres y madres y obtener únicos
    familiares_unicos = pd.concat([cedulas_padres, cedulas_madres]).drop_duplicates()
    total_familiares = len(familiares_unicos)

    # Obtener datos de ingresos unificados
    df_ingresos = datos_filtrados.get("Ingresos", pd.DataFrame())

    if df_ingresos.empty:
        return 0, total_familiares, total_hogares, 0

    # Filtrar por año 2025 y mes 6 (el último mes para determinar empleo actual)
    df_ingresos_mes6 = df_ingresos[
        (df_ingresos["anio"] == 2025) & (df_ingresos["mes"] == 6)
    ]

    # Obtener identificaciones únicas de familiares que trabajan en mes 6
    familiares_activos = df_ingresos_mes6["identificacion"].drop_duplicates()

    # Filtrar solo los familiares que pertenecen al periodo específico
    activos_en_periodo = familiares_activos[familiares_activos.isin(familiares_unicos)]

    activos = len(activos_en_periodo)
    desempleados = total_familiares - activos

    # Calcular Hogares con Trabajo (estudiantes donde al menos un familiar trabaja)
    # Obtener identificaciones de estudiantes cuyos familiares trabajan
    df_ingresos_estudiantes = df_ingresos_mes6[
        df_ingresos_mes6["identificacion"].isin(familiares_unicos)
    ]

    # Para cada estudiante, verificar si al menos un familiar trabaja
    estudiantes_con_trabajo = set()
    for _, row in df_ingresos_estudiantes.iterrows():
        # Buscar el estudiante que corresponde a este familiar
        estudiante_row = universo_periodo[
            (universo_periodo["ced_padre"] == row["identificacion"])
            | (universo_periodo["ced_madre"] == row["identificacion"])
        ]
        if not estudiante_row.empty:
            estudiantes_con_trabajo.update(estudiante_row["identificacion"].tolist())

    hogares_con_trabajo = len(estudiantes_con_trabajo)

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

    if len(salarios) == 0:
        return None

    # Crear boxplot
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=salarios,
            name="Salarios",
            boxpoints="outliers",
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


def mostrar_tarjetas_quintiles(quintiles_dict):
    """
    Muestra las 5 tarjetas de quintiles con diferentes colores

    Args:
        quintiles_dict: Diccionario con porcentajes por quintil
    """
    # Colores para cada quintil
    colores_quintiles = [
        COLORES["rojo"],  # Quintil 1 - Más bajo
        COLORES["naranja"],  # Quintil 2
        COLORES["amarillo"],  # Quintil 3 - Medio
        COLORES["cyan"],  # Quintil 4
        COLORES["verde"],  # Quintil 5 - Más alto
    ]

    st.write("**📊 Distribución por Quintiles de Ingresos**")

    if not quintiles_dict:
        st.info("No hay datos de quintiles disponibles para este periodo")
        return

    # Crear 5 columnas para los quintiles
    cols = st.columns(5)

    for i in range(5):
        quintil_key = f"Quintil {i+1}"
        percentage = quintiles_dict.get(quintil_key, 0)

        with cols[i]:
            tarjeta_simple(quintil_key, f"{percentage:.1f}%", colores_quintiles[i])


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


# Configuración de la página
st.set_page_config(page_title="Empleos e Ingresos", page_icon="💼", layout="wide")

# Título principal
st.title("💼 Empleos e Ingresos")

# Cargar datos
df_vulnerabilidad = cargar_datos_vulnerabilidad()

# Inicializar filtros
inicializar_filtros(df_vulnerabilidad)

# Mostrar filtros en la página principal
mostrar_filtros_en_pagina()

# Aplicar filtros
datos_filtrados = aplicar_filtros(df_vulnerabilidad)

# Obtener el grupo seleccionado
grupo_seleccionado = st.session_state.grupo_interes

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
                mostrar_tarjetas_quintiles(quintiles)

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
                    tarjeta_simple("Familiares Activos", activos, COLORES["verde"])
                with col_desempleados:
                    tarjeta_simple(
                        "Familiares Desempleados", desempleados, COLORES["rojo"]
                    )

                # Separador
                st.markdown("---")

                # Mostrar tarjetas de hogares
                col_total_hogares, col_hogares_trabajo = st.columns(2)
                with col_total_hogares:
                    tarjeta_simple("Total Hogares", total_hogares, COLORES["azul"])
                with col_hogares_trabajo:
                    tarjeta_simple(
                        "Hogares con Trabajo", hogares_con_trabajo, COLORES["cyan"]
                    )

                # Agregar boxplot de salarios
                st.markdown("---")
                st.write("**💰 Distribución de Salarios Familiares**")
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
                mostrar_tarjetas_quintiles(quintiles)

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
                mostrar_tarjetas_quintiles(quintiles)

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
                    tarjeta_simple("Familiares Activos", activos, COLORES["verde"])
                with col_desempleados:
                    tarjeta_simple(
                        "Familiares Desempleados", desempleados, COLORES["rojo"]
                    )

                # Separador
                st.markdown("---")

                # Mostrar tarjetas de hogares
                col_total_hogares, col_hogares_trabajo = st.columns(2)
                with col_total_hogares:
                    tarjeta_simple("Total Hogares", total_hogares, COLORES["azul"])
                with col_hogares_trabajo:
                    tarjeta_simple(
                        "Hogares con Trabajo", hogares_con_trabajo, COLORES["cyan"]
                    )

                # Agregar boxplot de salarios
                st.markdown("---")
                st.write("**💰 Distribución de Salarios Familiares**")
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
                mostrar_tarjetas_quintiles(quintiles)

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

    # Mostrar periodo también para Enrollment
    if periodos:
        st.write(f"### {periodos[0]}")

        # Calcular métricas de empleo para familiares de enrollment
        activos, desempleados, total_hogares, hogares_con_trabajo = (
            calcular_empleo_familiares(datos_filtrados, periodos[0])
        )

        # Mostrar tarjetas de familiares
        col_activos, col_desempleados = st.columns(2)
        with col_activos:
            tarjeta_simple("Familiares Activos", activos, COLORES["verde"])
        with col_desempleados:
            tarjeta_simple("Familiares Desempleados", desempleados, COLORES["rojo"])

        # Separador
        st.markdown("---")

        # Mostrar tarjetas de hogares
        col_total_hogares, col_hogares_trabajo = st.columns(2)
        with col_total_hogares:
            tarjeta_simple("Total Hogares", total_hogares, COLORES["azul"])
        with col_hogares_trabajo:
            tarjeta_simple("Hogares con Trabajo", hogares_con_trabajo, COLORES["cyan"])

        # Agregar boxplot de salarios
        st.markdown("---")
        st.write("**💰 Distribución de Salarios Familiares**")
        fig = crear_boxplot_salarios(datos_filtrados, periodos[0], grupo_seleccionado)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos de salarios disponibles para este periodo")

        # Agregar tarjetas de quintiles
        st.markdown("---")
        quintiles = calcular_quintiles(datos_filtrados, periodos[0], grupo_seleccionado)
        mostrar_tarjetas_quintiles(quintiles)

        # Agregar diagrama de Sankey para Enrollment
        st.markdown("---")
        st.write("**🔄 Transiciones de Tipo de Empleo**")
        fig_sankey = crear_diagrama_sankey(
            datos_filtrados, periodos[0], grupo_seleccionado
        )
        if fig_sankey:
            st.plotly_chart(fig_sankey, use_container_width=True)
        else:
            st.info("No hay datos suficientes para mostrar transiciones de empleo")
    else:
        st.write("No hay periodos disponibles para este grupo")
