import numpy as np
from scipy.stats import bootstrap
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.multivariate.manova import MANOVA

def build_cell_samples(
    df: pd.DataFrame,
    group_cols,                 # lista de factores, p.ej. ['sex', 'disease_stage']
    y_cols,                     # str (ANOVA) o list[str] (MANOVA)
    multivariate: bool
):
    """
    Crea una columna 'cell' (clave de celda), agrupa y devuelve:
      - cell_keys: lista ordenada (strings) con la etiqueta de cada celda ("F|I", "M|II", ...)
      - samples:   lista de arrays por celda:
          * ANOVA:  1D (n_i,) con la DV
          * MANOVA: 2D (n_i × p) con las DVs en el orden de y_cols
      - splitter:  función auxiliar para convertir una key -> dict{factor: nivel}
                   (la usamos luego para reconstruir el DataFrame)
    """
    if not isinstance(group_cols, (list, tuple)):
        group_cols = [group_cols]

    # 1) Creamos la clave de celda como string "A|B|C" (rápido y estable)
    #    Nota: si los factores ya son 'category', se respetará su orden interno.
    df = df.copy()
    df["cell"] = df[group_cols].astype(str).agg("|".join, axis=1)

    # 2) Orden determinista de celdas: por el orden alfabético de la clave 'cell'
    #    (si necesitas un orden específico de niveles, haz los factores 'category' con categorías ordenadas)
    groups = list(df.groupby("cell", sort=True))

    # 3) Armamos listas de claves y muestras por celda
    cell_keys = [name for name, _ in groups]
    if multivariate:
        if isinstance(y_cols, str):
            raise ValueError("Para MANOVA, y_cols debe ser list[str].")
        samples = [g[y_cols].to_numpy() for _, g in groups]          # (n_i × p)
    else:
        if isinstance(y_cols, (list, tuple)):
            raise ValueError("Para ANOVA, y_cols debe ser str (una DV).")
        samples = [g[y_cols].to_numpy() for _, g in groups]          # (n_i,)

    # 4) Pequeña utilidad para mapear "F|III" -> {'sex':'F', 'disease_stage':'III'}
    def splitter(key: str) -> dict:
        vals = key.split("|")
        return {c: v for c, v in zip(group_cols, vals)}

    return cell_keys, samples, splitter


def rebuild_df_from_bootstrap(
    cell_keys,                   # lista de strings "A|B|..."
    boot_arrays,                 # lista de arrays remuestreados por celda (en el mismo orden)
    group_cols,                  # lista de factores
    y_cols                       # str (ANOVA) o list[str] (MANOVA)
) -> pd.DataFrame:
    """
    Reconstruye un DataFrame a partir de:
      - cell_keys: p.ej., ["F|I", "F|II", "M|I", ...]
      - boot_arrays: datos remuestreados de cada celda (en el mismo orden)
      - group_cols: nombres de factores
      - y_cols: str (ANOVA) o list[str] (MANOVA)
    """
    if not isinstance(group_cols, (list, tuple)):
        group_cols = [group_cols]

    # 1) Calculamos los tamaños por celda (n_i) para repetir las etiquetas
    sizes = [arr.shape[0] for arr in boot_arrays]                 # longitudes de cada muestra
    total_n = int(np.sum(sizes))                                  # total de filas

    # 2) Construimos columnas de factores replicando niveles según n_i
    #    - Para cada celda "A|B", dividimos por '|' y agregamos 'A' repetida n_i veces a la columna del factor 1,
    #      'B' repetida n_i veces a la columna del factor 2, etc.
    factor_cols = {c: np.empty(total_n, dtype=object) for c in group_cols}
    pos = 0
    for key, n_i in zip(cell_keys, sizes):
        vals = key.split("|")                                     # ['F', 'I'] por ejemplo
        for j, c in enumerate(group_cols):
            factor_cols[c][pos:pos+n_i] = vals[j]
        pos += n_i

    # 3) Unimos los datos de respuesta (concatenados) con los factores
    if isinstance(y_cols, str):                                   # ANOVA: una DV
        y_concat = np.concatenate(boot_arrays, axis=0)            # (n_total,)
        df_b = pd.DataFrame({y_cols: y_concat, **factor_cols})
    else:                                                         # MANOVA: varias DVs
        Y_concat = np.vstack(boot_arrays)                         # (n_total × p)
        df_b = pd.DataFrame({**{d: Y_concat[:, j] for j, d in enumerate(y_cols)}, **factor_cols})

    return df_b
# =============================================================================
# Utilidad: construir ÍNDICES por celda (para remuestrear filas completas)
# =============================================================================



def build_cell_index_samples(df: pd.DataFrame, group_cols):
    """
    Devuelve:
      - cell_keys: lista de etiquetas de celda (e.g. 'F|I', 'M|III', ...)
      - cell_index_arrays: lista de arrays 1D con los ÍNDICES de filas por celda.
    Notas:
      * No modifica el df.
      * Soporta tamaños desbalanceados entre celdas.
    """
    if not isinstance(group_cols, (list, tuple)):
        group_cols = [group_cols]

    # Clave de celda como string "A|B|..." para agrupar
    gkey = df[group_cols].astype(str).agg("|".join, axis=1)

    # Agrupar y recolectar índices por celda (orden determinista por clave)
    groups = list(df.groupby(gkey, sort=True))
    cell_keys = [name for name, _ in groups]
    cell_index_arrays = [g.index.to_numpy() for _, g in groups]

    return cell_keys, cell_index_arrays

# =============================================================================
# -----------------------------------------------------------------------------
# Estas funciones NO hacen bootstrap. Solo calculan el estadístico “observado”
# con el pipeline habitual de statsmodels. Luego el bootstrap re-usará esto
# en cada réplica remuestreada.
# =============================================================================

def anova_F_observado(df: pd.DataFrame, formula: str, typ: int = 2) -> pd.Series:
    """
    Calcula la ANOVA clásica y devuelve una Serie con los F observados por término.
    - df: DataFrame con la DV y los factores
    - formula: 'DV ~ C(factor1)*C(factor2)+...' (misma notación que ya usas)
    - typ: 2 o 3 (sumas de cuadrados tipo II o III)
    Devuelve:
      Serie pandas: índice = término del modelo (e.g., 'C(sex)', 'C(disease_stage)', 'C(sex):C(disease_stage)'),
      valores = estadísticos F (sin incluir 'Residual').
    """
    mod = smf.ols(formula, data=df).fit()     # Ajusta OLS con la fórmula dada
    aov = anova_lm(mod, typ=typ)              # Obtiene la tabla ANOVA (SS, MS, F, p, etc.)
    F = aov['F'].dropna()                     # Quita filas sin F (p.ej., 'Residual'); quedas solo con términos
    return F


# =============================================================================
# MANOVA extendida: devuelve todos los tests (Pillai, Wilks, HL, Roy)
# =============================================================================

def manova_stats_todos(
    df: pd.DataFrame,
    dvs: list[str],
    formula_rhs: str
) -> pd.DataFrame:
    """
    Ejecuta MANOVA y devuelve un DataFrame con todos los tests por término:
      - Columnas: 'Pillai', 'Wilks', 'Hotelling-Lawley', 'Roy'
      - Filas: términos del modelo (sin 'Intercept')
    """
    # Construimos la fórmula general 'y1 + y2 + ... ~ factores'
    formula = ' + '.join(dvs) + ' ~ ' + formula_rhs

    # Ejecutamos MANOVA con statsmodels
    mv = MANOVA.from_formula(formula, data=df).mv_test()
    res = mv.results

    # Extraemos los cuatro tests por término (sin el intercepto)
    terms = [t for t in res.keys() if t.lower() != 'intercept']
    data = []
    for term in terms:
        stat = res[term]['stat']
        data.append([
            stat.loc["Pillai's trace", 'Value'],
            stat.loc["Wilks' lambda", 'Value'],
            stat.loc["Hotelling-Lawley trace", 'Value'],
            stat.loc["Roy's greatest root", 'Value']
        ])

    # Organizamos todo en un DataFrame bonito
    df_stats = pd.DataFrame(
        data,
        index=terms,
        columns=['Pillai', 'Wilks', 'Hotelling-Lawley', 'Roy']
    )
    return df_stats
# =============================================================================
# =============================================================================
# Bootstrap ANOVA (Percentil / BCa) 
# -----------------------------------------------------------------------------
# Requiere que ya estén definidas:
#   - build_cell_samples(df, group_cols, y_cols, multivariate=False)
#   - rebuild_df_from_bootstrap(cell_keys, boot_arrays, group_cols, y_cols)
#   - anova_F_observado(df, formula, typ)
# =============================================================================



def bootstrap_anova_percentil_bca(
    df,
    formula,            # ej: "IL6_pg_ml ~ C(sex)*C(disease_stage) + age_years + disease_duration_months"
    group_cols,         # ej: ['sex','disease_stage'] (definen las celdas del diseño)
    n_resamples=2000,   # B
    typ=2,              # SS tipo II o III (lo mismo que usas en clase)
    method="BCa",       # "percentile" o "BCa" (recomendado BCa)
    seed=0
):
    # ------------------------------
    # 0) Preparación básica
    # ------------------------------
    if not isinstance(group_cols, (list, tuple)):
        group_cols = [group_cols]
    rng = np.random.default_rng(seed)

    # Nombre de la DV (lado izquierdo de la fórmula), ej. "IL6_pg_ml"
    lhs = formula.split('~')[0].strip()

    # ------------------------------
    # 1) Estadístico observado (ANOVA normal con statsmodels)
    # ------------------------------
    F_obs = anova_F_observado(df, formula=formula, typ=typ)  # Serie: términos -> F
    terms = F_obs.index.tolist()                              # guardamos el orden de términos

    # ------------------------------
    # 2) Armamos las MUESTRAS por CELDA con tus utilidades
    #    (una array 1D por celda, en el mismo orden de cell_keys)
    # ------------------------------
    cell_keys, cell_samples, _ = build_cell_samples(
        df=df,
        group_cols=group_cols,
        y_cols=lhs,
        multivariate=False
    )

    # ------------------------------
    # 3) Función-estadístico para SciPy:
    #    - Recibe una TUPLA con los arrays remuestreados por celda (en el mismo orden de cell_keys)
    #    - Reconstruye un df con rebuild_df_from_bootstrap
    #    - Calcula ANOVA y devuelve un vector con F en el orden 'terms'
    # ------------------------------
    def stat_func(*cell_arrays):
        df_b = rebuild_df_from_bootstrap(
            cell_keys=cell_keys,
            boot_arrays=list(cell_arrays),
            group_cols=group_cols,
            y_cols=lhs
        )
        F_b = anova_F_observado(df_b, formula=formula, typ=typ)
        # Vectoriza en el orden de 'terms' (si un término no aparece, usamos 0.0 para robustez)
        return np.array([F_b.get(t, 0.0) for t in terms], dtype=float)

    # ------------------------------
    # 4) SciPy bootstrap (Percentil/BCa) — remuestreo ESTRATIFICADO por celdas
    # ------------------------------
    res = bootstrap(
        data=tuple(cell_samples),   # una muestra (array 1D) por celda
        statistic=stat_func,        # devuelve vector con F por término
        vectorized=False,           # nuestra función no es vectorizada
        paired=True,                # estratifica: remuestrea cada celda por separado
        n_resamples=n_resamples,    # B
        method=method,              # "percentile" o "BCa"
        random_state=rng
    )

    # ------------------------------
    # 5) Empaquetar resultados: distribución bootstrap y p_boot
    # ------------------------------
    boot_mat = res.bootstrap_distribution      # matriz (B × n_terms)
    boot_dist = {t: boot_mat[:, i] for i, t in enumerate(terms)}  # dict término->array F*
    p_boot = {t: np.mean(boot_dist[t] >= F_obs[t]) for t in terms}  # cola derecha

    return F_obs, p_boot, boot_dist, res
# =============================================================================
def bootstrap_anova_percentil_bca(
    df,
    formula,            # ej: "IL6_pg_ml ~ C(sex)*C(disease_stage) + age_years + disease_duration_months"
    group_cols,         # ej: ['sex','disease_stage']
    n_resamples=2000,
    typ=2,
    method="BCa",
    seed=0
):
    """
    ANOVA bootstrap estratificado por celdas remuestreando ÍNDICES de fila.
    Esto preserva DV, factores y COVARIABLES completas en cada réplica.
    Usa paired=False y axis=0 para permitir tamaños de celda desbalanceados.
    """
    if not isinstance(group_cols, (list, tuple)):
        group_cols = [group_cols]
    rng = np.random.default_rng(seed)

    # 1) Estadístico observado (tabla ANOVA clásica)
    mod_obs = smf.ols(formula, data=df).fit()
    aov_obs = anova_lm(mod_obs, typ=typ)
    F_obs = aov_obs['F'].dropna()
    terms = F_obs.index.tolist()

    # 2) ÍNDICES por celda (no datos). Cada array puede tener tamaño distinto (desbalanceado)
    cell_keys, cell_index_arrays = build_cell_index_samples(df, group_cols)

    # 3) Función-estadístico: recibe arrays de índices remuestreados y arma df_b con filas completas
    def stat_func(*boot_index_arrays):
        # Concatenamos los índices remuestreados de todas las celdas
        take = np.concatenate(boot_index_arrays, axis=0)
        # Subdataframe de la réplica con TODAS las columnas necesarias (DV, factores, covariables)
        df_b = df.loc[take]
        # Recalcular ANOVA
        m_b = smf.ols(formula, data=df_b).fit()
        a_b = anova_lm(m_b, typ=typ)
        # Vector de F en el mismo orden de 'terms' (si falta, 0.0)
        return np.array([a_b.loc[t, 'F'] if t in a_b.index else 0.0 for t in terms], dtype=float)

    # 4) SciPy bootstrap: remuestrear por filas en cada celda (longitudes distintas OK)
    res = bootstrap(
        data=tuple(cell_index_arrays),  # tuple de arrays de ÍNDICES (uno por celda)
        statistic=stat_func,
        vectorized=False,
        paired=False,                   # celdas pueden tener tamaños distintos
        axis=0,                         # remuestrea a lo largo de las filas (índices)
        n_resamples=n_resamples,
        method=method,
        random_state=rng
    )

    # 5) Distribución y p_boot
    boot_mat = res.bootstrap_distribution  # (B × n_terms)
    boot_dist = {t: boot_mat[:, i] for i, t in enumerate(terms)}
    p_boot = {t: np.mean(boot_dist[t] >= F_obs[t]) for t in terms}

    return F_obs, p_boot, boot_dist, res

# =============================================================================
def bootstrap_manova_percentil_bca(
    df,
    dvs,                 # ['IL6_pg_ml','CRP_mg_L','TNFa_pg_ml']
    formula_rhs,         # 'C(sex)*C(disease_stage) + age_years + disease_duration_months'
    group_cols,          # ['sex','disease_stage']
    n_resamples=2000,
    method="BCa",
    seed=0
):
    """
    MANOVA bootstrap estratificado por celdas remuestreando ÍNDICES de fila.
    Preserva DVs, factores y covariables completas en cada réplica.
    Devuelve p_boot para Pillai, Wilks, HL y Roy por término.
    """
    if not isinstance(group_cols, (list, tuple)):
        group_cols = [group_cols]
    rng = np.random.default_rng(seed)
    test_order = ['Pillai', 'Wilks', 'Hotelling-Lawley', 'Roy']

    # 1) Observados (tabla términos × tests)
    obs_df = manova_stats_todos(df, dvs=dvs, formula_rhs=formula_rhs)
    terms = obs_df.index.tolist()
    K = len(terms)
    stat_obs = {tname: obs_df[tname].to_dict() for tname in test_order}

    # 2) ÍNDICES por celda
    cell_keys, cell_index_arrays = build_cell_index_samples(df, group_cols)

    # 3) Función-estadístico: arma df_b con filas completas y calcula MANOVA (todos los tests)
    def stat_func(*boot_index_arrays):
        take = np.concatenate(boot_index_arrays, axis=0)
        df_b = df.loc[take]
        stats_b = manova_stats_todos(df_b, dvs=dvs, formula_rhs=formula_rhs)  # términos × tests
        # Vector concatenado en orden fijo por test y términos
        parts = [stats_b.reindex(index=terms)[col].fillna(0.0).to_numpy()
                 for col in test_order]
        return np.concatenate(parts, axis=0).astype(float)

    # 4) SciPy bootstrap: permitir tamaños distintos entre celdas
    res = bootstrap(
        data=tuple(cell_index_arrays),  # tuple de arrays de ÍNDICES (uno por celda)
        statistic=stat_func,
        vectorized=False,
        paired=False,
        axis=0,
        n_resamples=n_resamples,
        method=method,
        random_state=rng
    )

    # 5) Reorganizar distribución y calcular p_boot (colas correctas)
    mat = res.bootstrap_distribution  # (B × 4K)
    boot_dist = {
        'Pillai': {terms[i]: mat[:, i]           for i in range(0,      K)},
        'Wilks':  {terms[i]: mat[:, K + i]       for i in range(0,      K)},
        'HL':     {terms[i]: mat[:, 2*K + i]     for i in range(0,      K)},
        'Roy':    {terms[i]: mat[:, 3*K + i]     for i in range(0,      K)},
    }
    p_boot = {
        'Pillai': {t: np.mean(boot_dist['Pillai'][t] >= stat_obs['Pillai'][t]) for t in terms},
        'HL':     {t: np.mean(boot_dist['HL'][t]     >= stat_obs['Hotelling-Lawley'][t]) for t in terms},
        'Roy':    {t: np.mean(boot_dist['Roy'][t]    >= stat_obs['Roy'][t]) for t in terms},
        'Wilks':  {t: np.mean(boot_dist['Wilks'][t]  <= stat_obs['Wilks'][t]) for t in terms},
    }

    return terms, stat_obs, p_boot, boot_dist, res
# =============================================================================
# =============================================================================
# ANOVA bootstrap (estratificado por celdas) — desde cero
# -----------------------------------------------------------------------------
# • Siempre estratifica por los factores en `group_cols` (sin opción global).
# • Remuestrea FILAS completas dentro de cada celda (mantiene DV, factores y covariables).
# • Recalcula ANOVA (statsmodels) en cada réplica.
# • Devuelve: F observado, distribución F*, p_boot y CIs percentil; además un resumen bonito.
# =============================================================================

def anova_bootstrap_stratified(
    df: pd.DataFrame,
    formula: str,                   # ej: "IL6_pg_ml ~ C(sex)*C(disease_stage) + age_years + disease_duration_months"
    group_cols,                     # ej: ['sex','disease_stage']  (OBLIGATORIO → define celdas)
    B: int = 2000,                  # nº de réplicas bootstrap
    typ: int = 2,                   # SS tipo II (o 3 si lo prefieres)
    alpha: float = 0.05,            # para cuantiles/IC
    seed: int | None = 0,           # semilla reproducible
    progress: bool = False          # imprime progreso cada ~10%
):
    """
    ANOVA con bootstrap estratificado por celdas definidas por `group_cols`.

    Pasos:
      1) Calcula ANOVA observado (F por término).
      2) Construye celdas (combinaciones de niveles) en `group_cols`.
      3) Para b=1..B, remuestrea con reemplazo DENTRO de cada celda (mismo tamaño por celda),
         arma df_b con esas filas completas y recalcula ANOVA → F*_b por término.
      4) p_boot = P(F* ≥ F_obs) por término.  IC percentil 95% de F*.

    Retorna:
      F_obs   : pd.Series (términos → F observado)
      F_star  : dict[str, np.ndarray]  (término → array (B,) de F*)
      p_boot  : dict[str, float]
      ci      : dict[str, tuple(low, high)]  (IC percentil de F*)
      summary : pd.DataFrame con columnas: [F_obs, p_boot, q_(1-alpha), CI_low, CI_high]
    """
    # --- Validaciones básicas ---
    if group_cols is None or (isinstance(group_cols, (list, tuple)) and len(group_cols) == 0):
        raise ValueError("Debes especificar al menos un factor en `group_cols` para estratificar.")
    if not isinstance(group_cols, (list, tuple)):
        group_cols = [group_cols]
    rng = np.random.default_rng(seed)

    # --- 1) ANOVA observado (pipeline 'normal') ---
    mod = smf.ols(formula, data=df).fit()
    aov = anova_lm(mod, typ=typ)
    F_obs = aov["F"].dropna()                 # Serie: términos -> F (sin Residual/NaN)
    terms = F_obs.index.tolist()
    n_terms = len(terms)

    # --- 2) Celdas: índices de filas por combinación de niveles ---
    #     Clave de celda "A|B|..." y agrupación ordenada para orden determinista.
    cell_key = df[group_cols].astype(str).agg("|".join, axis=1)
    groups = list(df.groupby(cell_key, sort=True))
    cell_index_arrays = [g.index.to_numpy() for _, g in groups]

    # Sanidad: que ninguna celda esté vacía
    if any(len(idx) == 0 for idx in cell_index_arrays):
        raise ValueError("Hay celdas vacías en el diseño. Revisa `group_cols` y niveles.")

    # --- 3) Bucle bootstrap (remuestreo dentro de cada celda) ---
    F_mat = np.full((B, n_terms), np.nan, dtype=float)
    for b in range(B):
        # 3.1) para cada celda, sampleo con reemplazo el MISMO tamaño original
        boot_idx_parts = [rng.choice(idx, size=len(idx), replace=True) for idx in cell_index_arrays]
        take = np.concatenate(boot_idx_parts, axis=0)

        # 3.2) sub-DataFrame de la réplica (filas completas: DV+factores+covariables)
        df_b = df.loc[take]

        # 3.3) ANOVA en la réplica
        try:
            m_b = smf.ols(formula, data=df_b).fit()
            a_b = anova_lm(m_b, typ=typ)
            # guardo F* en el orden de 'terms'; si falta algún término -> NaN
            F_mat[b, :] = [a_b.loc[t, "F"] if t in a_b.index else np.nan for t in terms]
        except Exception:
            # si statsmodels falla (singularidad), dejamos NaN para esta réplica
            pass

        if progress and (b + 1) % max(1, B // 10) == 0:
            print(f"{b+1}/{B} réplicas...", end="\r")

    # --- 4) Salidas: F*, p_boot, IC, y resumen ---
    F_star = {t: F_mat[:, i] for i, t in enumerate(terms)}
    # p_boot (cola derecha) e IC percentil ignorando NaN
    p_boot = {t: float(np.mean(F_star[t] >= F_obs[t])) for t in terms}
    ci = {
        t: (float(np.nanpercentile(F_star[t], 100*alpha/2)),
            float(np.nanpercentile(F_star[t], 100*(1 - alpha/2))))
        for t in terms
    }
    q_1mA = {t: float(np.nanpercentile(F_star[t], 100*(1 - alpha))) for t in terms}

    summary = pd.DataFrame({
        "F_obs": F_obs,
        "p_boot": pd.Series(p_boot),
        f"q_{1-alpha:.2f}": pd.Series(q_1mA),
        "CI_low": pd.Series({t: ci[t][0] for t in terms}),
        "CI_high": pd.Series({t: ci[t][1] for t in terms}),
    }).loc[terms]

    return F_obs, F_star, p_boot, ci, summary
# =============================================================================
# =============================================================================
# ANOVA bootstrap (estratificado por celdas) — desde cero
# -----------------------------------------------------------------------------
# • Siempre estratifica por los factores en `group_cols` (sin opción global).
# • Remuestrea FILAS completas dentro de cada celda (mantiene DV, factores y covariables).
# • Recalcula ANOVA (statsmodels) en cada réplica.
# • Devuelve: F observado, distribución F*, p_boot y CIs percentil; además un resumen bonito.
# =============================================================================

def anova_bootstrap_stratified(
    df: pd.DataFrame,
    formula: str,                   # ej: "IL6_pg_ml ~ C(sex)*C(disease_stage) + age_years + disease_duration_months"
    group_cols,                     # ej: ['sex','disease_stage']  (OBLIGATORIO → define celdas)
    B: int = 2000,                  # nº de réplicas bootstrap
    typ: int = 2,                   # SS tipo II (o 3 si lo prefieres)
    alpha: float = 0.05,            # para cuantiles/IC
    seed: int | None = 0,           # semilla reproducible
    progress: bool = False          # imprime progreso cada ~10%
):
    """
    ANOVA con bootstrap estratificado por celdas definidas por `group_cols`.

    Pasos:
      1) Calcula ANOVA observado (F por término).
      2) Construye celdas (combinaciones de niveles) en `group_cols`.
      3) Para b=1..B, remuestrea con reemplazo DENTRO de cada celda (mismo tamaño por celda),
         arma df_b con esas filas completas y recalcula ANOVA → F*_b por término.
      4) p_boot = P(F* ≥ F_obs) por término.  IC percentil 95% de F*.

    Retorna:
      F_obs   : pd.Series (términos → F observado)
      F_star  : dict[str, np.ndarray]  (término → array (B,) de F*)
      p_boot  : dict[str, float]
      ci      : dict[str, tuple(low, high)]  (IC percentil de F*)
      summary : pd.DataFrame con columnas: [F_obs, p_boot, q_(1-alpha), CI_low, CI_high]
    """
    # --- Validaciones básicas ---
    if group_cols is None or (isinstance(group_cols, (list, tuple)) and len(group_cols) == 0):
        raise ValueError("Debes especificar al menos un factor en `group_cols` para estratificar.")
    if not isinstance(group_cols, (list, tuple)):
        group_cols = [group_cols]
    rng = np.random.default_rng(seed)

    # --- 1) ANOVA observado (pipeline 'normal') ---
    mod = smf.ols(formula, data=df).fit()
    aov = anova_lm(mod, typ=typ)
    F_obs = aov["F"].dropna()                 # Serie: términos -> F (sin Residual/NaN)
    terms = F_obs.index.tolist()
    n_terms = len(terms)

    # --- 2) Celdas: índices de filas por combinación de niveles ---
    #     Clave de celda "A|B|..." y agrupación ordenada para orden determinista.
    cell_key = df[group_cols].astype(str).agg("|".join, axis=1)
    groups = list(df.groupby(cell_key, sort=True))
    cell_index_arrays = [g.index.to_numpy() for _, g in groups]

    # Sanidad: que ninguna celda esté vacía
    if any(len(idx) == 0 for idx in cell_index_arrays):
        raise ValueError("Hay celdas vacías en el diseño. Revisa `group_cols` y niveles.")

    # --- 3) Bucle bootstrap (remuestreo dentro de cada celda) ---
    F_mat = np.full((B, n_terms), np.nan, dtype=float)
    for b in range(B):
        # 3.1) para cada celda, sampleo con reemplazo el MISMO tamaño original
        boot_idx_parts = [rng.choice(idx, size=len(idx), replace=True) for idx in cell_index_arrays]
        take = np.concatenate(boot_idx_parts, axis=0)

        # 3.2) sub-DataFrame de la réplica (filas completas: DV+factores+covariables)
        df_b = df.loc[take]

        # 3.3) ANOVA en la réplica
        try:
            m_b = smf.ols(formula, data=df_b).fit()
            a_b = anova_lm(m_b, typ=typ)
            # guardo F* en el orden de 'terms'; si falta algún término -> NaN
            F_mat[b, :] = [a_b.loc[t, "F"] if t in a_b.index else np.nan for t in terms]
        except Exception:
            # si statsmodels falla (singularidad), dejamos NaN para esta réplica
            pass

        if progress and (b + 1) % max(1, B // 10) == 0:
            print(f"{b+1}/{B} réplicas...", end="\r")

    # --- 4) Salidas: F*, p_boot, IC, y resumen ---
    F_star = {t: F_mat[:, i] for i, t in enumerate(terms)}
    # p_boot (cola derecha) e IC percentil ignorando NaN
    p_boot = {t: float(np.mean(F_star[t] >= F_obs[t])) for t in terms}
    ci = {
        t: (float(np.nanpercentile(F_star[t], 100*alpha/2)),
            float(np.nanpercentile(F_star[t], 100*(1 - alpha/2))))
        for t in terms
    }
    q_1mA = {t: float(np.nanpercentile(F_star[t], 100*(1 - alpha))) for t in terms}

    summary = pd.DataFrame({
        "F_obs": F_obs,
        "p_boot": pd.Series(p_boot),
        f"q_{1-alpha:.2f}": pd.Series(q_1mA),
        "CI_low": pd.Series({t: ci[t][0] for t in terms}),
        "CI_high": pd.Series({t: ci[t][1] for t in terms}),
    }).loc[terms]

    return F_obs, F_star, p_boot, ci, summary

# =============================================================================
def plot_bootstrap_anova_robusto(term, F_obs, boot_dist, p_boot):
    vals = np.asarray(boot_dist[term], float)
    obs  = float(F_obs[term])
    p    = float(p_boot[term])

    B = len(vals)
    nunique = np.unique(vals).size
    nzeros  = np.sum(np.isclose(vals, 0))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # — Histograma o gráfico de barras discreto si hay pocos únicos —
    if nunique <= 15:
        uniq, counts = np.unique(vals, return_counts=True)
        ax[0].bar(uniq, counts, width=0.08*(uniq.max()-uniq.min()+1) if uniq.size>1 else 0.5, edgecolor="k")
        ax[0].set_xlabel("F* (bootstrap)")
        ax[0].set_ylabel("Frecuencia")
    else:
        bins = max(10, min(50, int(np.sqrt(B))))
        ax[0].hist(vals, bins=bins, alpha=0.75, edgecolor='k')
        ax[0].set_xlabel("F* (bootstrap)")
        ax[0].set_ylabel("Frecuencia")

    ax[0].axvline(obs, color='red', linestyle='--', linewidth=2, label=f"F_obs={obs:.3f}")
    ax[0].legend()
    ax[0].set_title(f"Distribución bootstrap — {term}\np_boot={p:.4f}")

    # — ECDF (siempre informativa aunque haya empates) —
    x = np.sort(vals)
    y = np.arange(1, B+1)/B
    ax[1].plot(x, y)
    ax[1].axvline(obs, color='red', linestyle='--', linewidth=2)
    ax[1].set_xlabel("F* (bootstrap)")
    ax[1].set_ylabel("ECDF")
    ax[1].set_title("ECDF de F*")

    plt.tight_layout()
    plt.show()

# =============================================================================
# --- 1) Estadísticos OBSERVADOS (Welch) ---
def mean_diff(a, b, axis=0):
    # diferencia de medias (a - b)
    return np.mean(a, axis=axis) - np.mean(b, axis=axis)

def welch_t(a, b, axis=0):
    # t de Welch: (m1-m2) / sqrt(s1^2/n1 + s2^2/n2)
    a = np.asarray(a); b = np.asarray(b)
    m1 = np.mean(a, axis=axis); m2 = np.mean(b, axis=axis)
    v1 = np.var(a, axis=axis, ddof=1); v2 = np.var(b, axis=axis, ddof=1)
    n1 = a.shape[axis];            n2 = b.shape[axis]
    se = np.sqrt(v1/n1 + v2/n2)
    return (m1 - m2) / se

#=============================================================================

