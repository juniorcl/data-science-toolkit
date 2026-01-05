import pandas as pd
from scipy.stats import chi2_contingency


def chi_squared_monitoring(expected, actual, alpha=0.05, min_expected_freq=5):
    """
    Aplica o teste Qui-Quadrado para monitorar drift em uma variável categórica.

    Parameters
    ----------
    expected : pd.Series or array-like
        Distribuição de referência (ex: dados de treino)
    actual : pd.Series or array-like
        Distribuição atual (ex: dados em produção)
    alpha : float, default=0.05
        Nível de significância do teste
    min_expected_freq : int, default=5
        Frequência mínima esperada para validade do teste

    Returns
    -------
    dict
        Resultados do teste e métricas auxiliares
    """

    if not isinstance(expected, pd.Series):
        expected = pd.Series(expected)

    if not isinstance(actual, pd.Series):
        actual = pd.Series(actual)

    expected = expected.dropna()
    actual = actual.dropna()

    categories = sorted(set(expected) | set(actual))

    ref_counts = expected.value_counts().reindex(categories, fill_value=0)
    cur_counts = actual.value_counts().reindex(categories, fill_value=0)

    contingency_table = pd.DataFrame(
        {
            "reference": ref_counts,
            "current": cur_counts
        }
    ).T

    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    expected_df = pd.DataFrame(
        expected,
        index=contingency_table.index,
        columns=contingency_table.columns
    )

    valid_test = (expected_df >= min_expected_freq).all().all()

    alert = (p_value < alpha) and valid_test

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "drift_detected": bool(alert)
    }