# Plot text on a graph

def plot_text(ax: plt.Axes):
    """
    text on barplot
    """
    for p in ax.patches:
        percentage = '{:.1f}%'.format(p.get_height())
        ax.annotate(
            percentage,  # text
            # coordinate xy
            (p.get_x() + p.get_width() / 2., p.get_height()),
            # center
            ha='center',
            va='center',
            xytext=(0, 10),
            # offset point
            textcoords='offset points',
            fontsize=14)


def barplot_group_by_category(df_data: pd.DataFrame, col_main: str, col_group: str,
                              title: str) -> None:
    """
    build barplot with normalized data and graph data annotations

    highligting target variable by variable
    """

    plt.figure(figsize=(15, 6))

    data = (df_data.groupby([col_main])[col_group].value_counts(normalize=True).rename(
        'percentage').mul(100).reset_index().sort_values(col_main))

    ax = sns.barplot(x=col_main,
                     y="percentage",
                     hue=col_group,
                     data=data,
                     palette='rocket')

    for p in ax.patches:
        percentage = '{:.1f}%'.format(p.get_height())
        ax.annotate(
            percentage,
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='center',
            xytext=(0, 7),
            textcoords='offset points',
            fontsize=12)

    plt.title(title, fontsize=16)
    plt.ylabel('Percentage', fontsize=14)
    plt.xlabel(col_main, fontsize=14)
    plt.show()




def barplot_group_within_targetvar(df_data: pd.DataFrame, col_main: str, col_group: str,
                                   title: str) -> None:
    """
    build barplot with normalized data and graph data annotations

    highligting proportion of variable within target group
    """

    plt.figure(figsize=(15, 6))

    data = (df_data.groupby([col_group])[col_main].value_counts(normalize=True).rename(
        'percentage').mul(100).reset_index().sort_values(col_group))

    ax = sns.barplot(x=col_main,
                     y="percentage",
                     hue=col_group,
                     data=data,
                     palette='rocket')

    for p in ax.patches:
        percentage = '{:.1f}%'.format(p.get_height())
        ax.annotate(
            percentage,
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='center',
            xytext=(0, 7),
            textcoords='offset points',
            fontsize=12)

    plt.title(title, fontsize=16)
    plt.ylabel('Percentage', fontsize=14)
    plt.xlabel(col_main, fontsize=14)
    plt.show()



#%%
def barplot_group_within_targetvar(df_data: pd.DataFrame, col_main: str, col_group: str,
                  title: str) -> None:
    """
    build barplot with normalized data and graph data annotations

    highligting proportion of variable within target group
    """

    plt.figure(figsize=(15, 6))

    data = (df_data.groupby([col_group])[col_main].value_counts(normalize=True).rename(
            'percentage').mul(100).reset_index().sort_values(col_group))

    ax = sns.barplot(x=col_main,
                     y="percentage",
                     hue=col_group,
                     data=data,
                     palette='rocket')

    for p in ax.patches:
        percentage = '{:.1f}%'.format(p.get_height())
        ax.annotate(
            percentage,
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='center',
            xytext=(0, 7),
            textcoords='offset points',
            fontsize=12)

    plt.title(title, fontsize=16)
    plt.ylabel('Percentage', fontsize=14)
    plt.xlabel(col_main, fontsize=14)
    plt.show()

#%%
# checking for multicollinearity - VIF (Variance Inflation Factor)
# tells you how much the variance of a regression coefficient is inflated due to multicollinearity

def compute_vif(df):
    vif_data = []
    X = df.dropna()
    for i in range(X.shape[1]):
        y = X.iloc[:, i]
        X_other = X.drop(X.columns[i], axis=1)

        model = LinearRegression().fit(X_other, y)
        r2 = model.score(X_other, y)
        vif = 1 / (1 - r2) if r2 < 1 else float('inf')

        vif_data.append({'Variable': X.columns[i], 'VIF': vif})

    return pd.DataFrame(vif_data)
#%%
# chi-square test and Cramer's V functions for categorical variables association

def cramers_v(confusion_matrix):
    """Compute Cramér's V (strength of association) from a confusion matrix."""
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def categorical_assoc_with_churn(df, churn_col='Churn'):
    """
    For each categorical feature in df (excluding churn_col),
    run Chi-square test + Cramér's V against churn.
    """
    results = []

    # Loop over categorical columns
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col == churn_col:
            continue  # skip target

        contingency_table = pd.crosstab(df[churn_col], df[col])

        # Chi-square
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

        # Cramer's V
        cv = cramers_v(contingency_table)

        results.append({
            'Feature': col,
            'Chi2': chi2,
            'p_value': p,
            'Cramers_V': cv
        })

    return pd.DataFrame(results).sort_values('Cramers_V', ascending=False)



