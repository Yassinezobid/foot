import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration de la page
st.set_page_config( 
    page_title="SimuProfit - Business Plan Mini Football",
    page_icon="⚽",
    layout="wide"
)

# Fonction pour ajouter un style CSS personnalisé
def local_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2196F3;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
    }
    .positive-value {
        color: #28a745;
    }
    .negative-value {
        color: #dc3545;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .profit-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .chart-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# Titre principal de l'application
st.markdown('<p class="main-header">⚽ SimuProfit - Business Plan Mensuel Mini Football</p>', unsafe_allow_html=True)
st.markdown("### Simulez la rentabilité de vos terrains de mini football en quelques clics")

# Initialisation des variables de session si elles n'existent pas déjà
if 'prix_vente' not in st.session_state:
    # Liste des services avec leurs emojis
    services = {
        "Location 1 heure": "⚽",
        "Abonnement mensuel": "📅",
        "Tournois": "🏆",
        "Académie": "👨‍🏫",
        "Vente d'équipements": "👕",
        "Boissons": "🥤",
        "Snacks": "🍔"
    }
    
    # Initialisation des dictionnaires dans la session
    st.session_state.services = services
    st.session_state.prix_vente = {
        "Location 1 heure": 200.0,
        "Abonnement mensuel": 800.0,
        "Tournois": 1500.0,
        "Académie": 300.0,
        "Vente d'équipements": 150.0,
        "Boissons": 15.0,
        "Snacks": 25.0
    }
    
    st.session_state.cout_unitaire = {
        "Location 1 heure": 50.0,
        "Abonnement mensuel": 200.0,
        "Tournois": 500.0,
        "Académie": 100.0,
        "Vente d'équipements": 75.0,
        "Boissons": 5.0,
        "Snacks": 10.0
    }
    
    st.session_state.commandes_jour = {
        "Location 1 heure": 8,
        "Abonnement mensuel": 1,
        "Tournois": 0.2,
        "Académie": 1,
        "Vente d'équipements": 2,
        "Boissons": 20,
        "Snacks": 10
    }
    
    # Initialisation des paramètres d'activité
    st.session_state.jours_activite = 26
    st.session_state.taux_impot = 20.0
    st.session_state.nb_associes = 2
    st.session_state.nb_terrains = 1
    
    # Initialisation des charges mensuelles (exactement comme spécifié)
    st.session_state.charges_mensuelles = {
        "Loyer": 8000.0,
        "Électricité et eau": 3500.0,
        "Employés": 6000.0
    }
    
    # Initialisation des charges d'investissement (exactement comme spécifié)
    st.session_state.charges_investissement = {
        "Avance de terrain": 40000.0,
        "Construction": 150000.0,
        "Gazon": 250000.0,
        "Publicités": 20000.0,
        "Stock initial": 15000.0,
        "Social Media et App": 10000.0,
        "Caméras de surveillance": 5000.0,
        "Caméras de filmage": 10000.0,
        "Divers": 5000.0,
        "Création d'association": 5000.0
    }

# Dictionnaire des emojis pour les charges
charges_emojis = {
    "Loyer": "🏢",
    "Électricité et eau": "⚡",
    "Employés": "👨‍💼"
}

# Sidebar masquée mais utilisable si nécessaire
with st.sidebar:
    st.markdown("### ⚙️ Paramètres supplémentaires")
    st.markdown("Utilisez directement les tableaux principaux pour modifier les valeurs")

# Fonction pour calculer les indicateurs financiers
def calculer_indicateurs():
    # Calcul des revenus et coûts par service
    revenus_services = {}
    couts_services = {}
    marges_services = {}

    for service in st.session_state.services:
        # Pour les services de location, on multiplie par le nombre de terrains
        if service in ["Location 1 heure", "Abonnement mensuel", "Tournois", "Académie"]:
            revenus_services[service] = st.session_state.prix_vente[service] * st.session_state.commandes_jour[service] * st.session_state.jours_activite * st.session_state.nb_terrains
            couts_services[service] = st.session_state.cout_unitaire[service] * st.session_state.commandes_jour[service] * st.session_state.jours_activite * st.session_state.nb_terrains
        else:
            # Les autres services comme boissons, snacks ne dépendent pas directement du nombre de terrains
            revenus_services[service] = st.session_state.prix_vente[service] * st.session_state.commandes_jour[service] * st.session_state.jours_activite
            couts_services[service] = st.session_state.cout_unitaire[service] * st.session_state.commandes_jour[service] * st.session_state.jours_activite
        
        marges_services[service] = revenus_services[service] - couts_services[service]

    # Calcul des totaux
    revenu_brut = sum(revenus_services.values())
    cout_variable = sum(couts_services.values())
    cout_fixe = sum(st.session_state.charges_mensuelles.values())
    cout_total = cout_variable + cout_fixe
    benefice_brut = revenu_brut - cout_total
    impot = benefice_brut * (st.session_state.taux_impot / 100) if benefice_brut > 0 else 0
    profit_net = benefice_brut - impot
    profit_par_associe = profit_net / st.session_state.nb_associes if st.session_state.nb_associes > 0 else 0
    
    # Total des investissements (pour un terrain)
    total_investissement_un_terrain = sum(st.session_state.charges_investissement.values())
    
    # Pour le total d'investissement, multiplions par le nombre de terrains pour les éléments spécifiques aux terrains
    investissement_specifique_terrain = (
        st.session_state.charges_investissement["Avance de terrain"] +
        st.session_state.charges_investissement["Construction"] +
        st.session_state.charges_investissement["Gazon"] +
        st.session_state.charges_investissement["Caméras de surveillance"] +
        st.session_state.charges_investissement["Caméras de filmage"]
    )
    
    investissement_commun = (
        st.session_state.charges_investissement["Publicités"] +
        st.session_state.charges_investissement["Stock initial"] +
        st.session_state.charges_investissement["Social Media et App"] +
        st.session_state.charges_investissement["Divers"] +
        st.session_state.charges_investissement["Création d'association"]
    )
    
    total_investissement = investissement_specifique_terrain * st.session_state.nb_terrains + investissement_commun
    
    # Calcul du seuil de rentabilité
    if revenu_brut > 0:
        seuil_rentabilite = cout_fixe / (1 - (cout_variable / revenu_brut))
        marge_cout_variable = (1 - (cout_variable / revenu_brut)) * 100
    else:
        seuil_rentabilite = 0
        marge_cout_variable = 0
    
    # Calcul du ROI
    if total_investissement > 0 and profit_net > 0:
        roi_mensuel = profit_net / total_investissement * 100
        roi_annuel = roi_mensuel * 12
        temps_retour = total_investissement / profit_net
    else:
        roi_mensuel = 0
        roi_annuel = 0
        temps_retour = float('inf')
    
    return {
        'revenus_services': revenus_services,
        'couts_services': couts_services,
        'marges_services': marges_services,
        'revenu_brut': revenu_brut,
        'cout_variable': cout_variable,
        'cout_fixe': cout_fixe,
        'cout_total': cout_total,
        'benefice_brut': benefice_brut,
        'impot': impot,
        'profit_net': profit_net,
        'profit_par_associe': profit_par_associe,
        'total_investissement': total_investissement,
        'total_investissement_un_terrain': total_investissement_un_terrain,
        'seuil_rentabilite': seuil_rentabilite,
        'marge_cout_variable': marge_cout_variable,
        'roi_mensuel': roi_mensuel,
        'roi_annuel': roi_annuel,
        'temps_retour': temps_retour
    }

# Calculer les indicateurs financiers
indicateurs = calculer_indicateurs()

# Contenu principal
# 1. Affichage du résumé financier
st.markdown("## 💰 Résumé financier")
col_profit1, col_profit2, col_profit3 = st.columns(3)
with col_profit1:
    st.metric(label="Profit Net Total", value=f"{indicateurs['profit_net']:.2f} DH",
            delta=f"{indicateurs['profit_net']:.1f} DH" if indicateurs['profit_net'] > 0 else f"-{abs(indicateurs['profit_net']):.1f} DH")
with col_profit2:
    st.metric(label="Par Associé", value=f"{indicateurs['profit_par_associe']:.2f} DH")
with col_profit3:
    st.metric(label="ROI annuel", value=f"{indicateurs['roi_annuel']:.2f}%")

# 2. Paramètres d'activité généraux (dans un formulaire éditable)
st.markdown('<p class="sub-header">📆 Paramètres d\'activité</p>', unsafe_allow_html=True)
params_col1, params_col2, params_col3, params_col4 = st.columns(4)

with params_col1:
    jours_activite = st.number_input(
        "Nombre de jours d'activité par mois",
        min_value=1,
        max_value=31,
        value=st.session_state.jours_activite,
        step=1,
        key="jours_activite_input"
    )
    st.session_state.jours_activite = jours_activite

with params_col2:
    taux_impot = st.number_input(
        "Taux d'impôt (%)",
        min_value=0.0,
        max_value=50.0,
        value=st.session_state.taux_impot,
        step=0.5,
        key="taux_impot_input"
    )
    st.session_state.taux_impot = taux_impot

with params_col3:
    nb_associes = st.number_input(
        "Nombre d'associés",
        min_value=1,
        value=st.session_state.nb_associes,
        step=1,
        key="nb_associes_input"
    )
    st.session_state.nb_associes = nb_associes

with params_col4:
    nb_terrains = st.number_input(
        "Nombre de terrains",
        min_value=1,
        value=st.session_state.nb_terrains,
        step=1,
        key="nb_terrains_input"
    )
    st.session_state.nb_terrains = nb_terrains

# 3. Tableau de bord financier
st.markdown('<p class="sub-header">📊 Tableau de bord financier</p>', unsafe_allow_html=True)

# Visualisation du profit net
fig, ax = plt.subplots(figsize=(10, 6))
labels = ['Revenu brut', 'Coût total', 'Bénéfice brut', 'Impôt', 'Profit net']
values = [
    indicateurs['revenu_brut'],
    indicateurs['cout_total'],
    indicateurs['benefice_brut'],
    indicateurs['impot'],
    indicateurs['profit_net']
]

bars = ax.bar(labels, values)

# Coloriser les barres selon les valeurs positives/négatives
for i, bar in enumerate(bars):
    if values[i] < 0:
        bar.set_color('#dc3545')  # Rouge pour valeurs négatives
    else:
        bar.set_color('#28a745')  # Vert pour valeurs positives

plt.ylabel('Montant (DH)')
plt.title('Répartition financière mensuelle')

# Ajouter les valeurs au-dessus des barres
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 50,
            f'{height:.2f} DH', ha='center', va='bottom')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Affichage du graphique dans un container stylisé
with st.container():
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# Tableau résumé des indicateurs financiers
data_resume = {
    "Indicateur": ["Revenu brut mensuel", "Coût variable (services)", "Coût fixe (charges)",
                     "Coût total mensuel", "Bénéfice avant impôt", f"Impôt ({st.session_state.taux_impot}%)",
                     "Profit net mensuel", f"Profit par associé ({st.session_state.nb_associes})"],
    "Montant (DH)": [
        indicateurs['revenu_brut'],
        indicateurs['cout_variable'],
        indicateurs['cout_fixe'],
        indicateurs['cout_total'],
        indicateurs['benefice_brut'],
        indicateurs['impot'],
        indicateurs['profit_net'],
        indicateurs['profit_par_associe']
    ]
}

df_resume = pd.DataFrame(data_resume)
df_resume["Montant (DH)"] = df_resume["Montant (DH)"].apply(lambda x: f"{x:.2f} DH")

st.dataframe(df_resume, use_container_width=True)

# 4. Tableau détaillé des services (éditable)
st.markdown('<p class="sub-header">⚽ Détails par service</p>', unsafe_allow_html=True)

# Créer un DataFrame pour les services avec les colonnes éditables
services_data = []
for service in st.session_state.services:
    emoji = st.session_state.services[service]
    if service in ["Location 1 heure", "Abonnement mensuel", "Tournois", "Académie"]:
        service_label = f"{emoji} {service} (par terrain)"
    else:
        service_label = f"{emoji} {service}"
    services_data.append({
        "Service": service_label,
        "Service_key": service,  # Clé pour référence
        "Prix unitaire (DH)": st.session_state.prix_vente[service],
        "Coût unitaire (DH)": st.session_state.cout_unitaire[service],
        "Commandes/jour": st.session_state.commandes_jour[service],
        "Revenu mensuel (DH)": indicateurs['revenus_services'][service],
        "Coût mensuel (DH)": indicateurs['couts_services'][service],
        "Marge mensuelle (DH)": indicateurs['marges_services'][service],
    })

df_services = pd.DataFrame(services_data)

# Utiliser un formulaire pour la modification
with st.form(key="services_form"):
    # Table éditable pour les services
    for i, row in enumerate(services_data):
        st.markdown(f"#### {row['Service']}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prix = st.number_input(
                "Prix unitaire (DH)",
                min_value=0.0,
                value=float(row['Prix unitaire (DH)']),
                step=0.5,
                key=f"prix_{i}"
            )
        
        with col2:
            cout = st.number_input(
                "Coût unitaire (DH)",
                min_value=0.0,
                value=float(row['Coût unitaire (DH)']),
                step=0.1,
                key=f"cout_{i}"
            )
        
        with col3:
            commandes = st.number_input(
                "Commandes/jour",
                min_value=0.0,
                value=float(row['Commandes/jour']),
                step=0.1,
                key=f"commandes_{i}"
            )
        
        # Mise à jour des valeurs
        service_key = row['Service_key']
        st.session_state.prix_vente[service_key] = prix
        st.session_state.cout_unitaire[service_key] = cout
        st.session_state.commandes_jour[service_key] = commandes
        
        st.markdown("---")
    
    # Bouton pour soumettre les modifications
    submitted = st.form_submit_button("Mettre à jour les calculs")
    if submitted:
        st.success("Valeurs mises à jour! Les calculs ont été recalculés.")
        indicateurs = calculer_indicateurs()  # Recalculer les indicateurs

# Affichage des résultats calculés pour les services
# Recréer le DataFrame avec les valeurs mises à jour
services_data_updated = []
for service in st.session_state.services:
    emoji = st.session_state.services[service]
    marge_unitaire = st.session_state.prix_vente[service] - st.session_state.cout_unitaire[service]
    
    # Pour les services liés aux terrains, prendre en compte le nombre de terrains
    if service in ["Location 1 heure", "Abonnement mensuel", "Tournois", "Académie"]:
        service_label = f"{emoji} {service} (par terrain)"
        revenu_mensuel = st.session_state.prix_vente[service] * st.session_state.commandes_jour[service] * st.session_state.jours_activite * st.session_state.nb_terrains
        cout_mensuel = st.session_state.cout_unitaire[service] * st.session_state.commandes_jour[service] * st.session_state.jours_activite * st.session_state.nb_terrains
    else:
        service_label = f"{emoji} {service}"
        revenu_mensuel = st.session_state.prix_vente[service] * st.session_state.commandes_jour[service] * st.session_state.jours_activite
        cout_mensuel = st.session_state.cout_unitaire[service] * st.session_state.commandes_jour[service] * st.session_state.jours_activite
    
    marge_mensuelle = revenu_mensuel - cout_mensuel
    
    services_data_updated.append({
        "Service": service_label,
        "Prix unitaire (DH)": f"{st.session_state.prix_vente[service]:.2f} DH",
        "Coût unitaire (DH)": f"{st.session_state.cout_unitaire[service]:.2f} DH",
        "Marge unitaire (DH)": f"{marge_unitaire:.2f} DH",
        "Commandes/jour": st.session_state.commandes_jour[service],
        "Revenu mensuel (DH)": f"{revenu_mensuel:.2f} DH",
        "Coût mensuel (DH)": f"{cout_mensuel:.2f} DH",
        "Marge mensuelle (DH)": f"{marge_mensuelle:.2f} DH"
    })

# Ajouter une ligne de total
total_commands = sum(st.session_state.commandes_jour.values())
total_revenue = sum(indicateurs['revenus_services'].values())
total_costs = sum(indicateurs['couts_services'].values())
total_margins = sum(indicateurs['marges_services'].values())

services_data_updated.append({
    "Service": "📊 TOTAL",
    "Prix unitaire (DH)": "-",
    "Coût unitaire (DH)": "-",
    "Marge unitaire (DH)": "-",
    "Commandes/jour": total_commands,
    "Revenu mensuel (DH)": f"{total_revenue:.2f} DH",
    "Coût mensuel (DH)": f"{total_costs:.2f} DH",
    "Marge mensuelle (DH)": f"{total_margins:.2f} DH"
})

df_services_updated = pd.DataFrame(services_data_updated)
st.dataframe(df_services_updated, use_container_width=True)

# 5. Tableau des charges mensuelles (éditable)
st.markdown('<p class="sub-header">💸 Détail des charges mensuelles</p>', unsafe_allow_html=True)

with st.form(key="charges_form"):
    # Table éditable pour les charges
    charges_data = []
    
    # Utiliser des colonnes pour organiser les champs de formulaire
    col1, col2 = st.columns(2)
    charges_keys = list(st.session_state.charges_mensuelles.keys())
    
    half = len(charges_keys) // 2 + len(charges_keys) % 2
    
    with col1:
        for i, charge in enumerate(charges_keys[:half]):
            emoji = charges_emojis.get(charge, "📝")
            montant = st.number_input(
                f"{emoji} {charge} (DH)",
                min_value=0.0,
                value=st.session_state.charges_mensuelles[charge],
                step=100.0,
                format="%.2f",
                key=f"charge_{i}"
            )
            st.session_state.charges_mensuelles[charge] = montant
            charges_data.append({
                "Charge": f"{emoji} {charge}",
                "Montant (DH)": f"{montant:.2f} DH"
            })
    
    with col2:
        for i, charge in enumerate(charges_keys[half:]):
            emoji = charges_emojis.get(charge, "📝")
            montant = st.number_input(
                f"{emoji} {charge} (DH)",
                min_value=0.0,
                value=st.session_state.charges_mensuelles[charge],
                step=100.0,
                format="%.2f",
                key=f"charge_{i + half}"
            )
            st.session_state.charges_mensuelles[charge] = montant
            charges_data.append({
                "Charge": f"{emoji} {charge}",
                "Montant (DH)": f"{montant:.2f} DH"
            })
    
    # Bouton pour soumettre les modifications
    charges_submitted = st.form_submit_button("Mettre à jour les charges")
    if charges_submitted:
        st.success("Charges mises à jour! Les calculs ont été recalculés.")
        indicateurs = calculer_indicateurs()  # Recalculer les indicateurs

# Ajouter une ligne de total pour les charges
total_charges = sum(st.session_state.charges_mensuelles.values())
charges_data.append({
    "Charge": "📊 TOTAL",
    "Montant (DH)": f"{total_charges:.2f} DH"
})

df_charges = pd.DataFrame(charges_data)
st.dataframe(df_charges, use_container_width=True)

# 6. Tableau des charges d'investissement (éditable)
st.markdown('<p class="sub-header">🏗️ Charges d\'investissement</p>', unsafe_allow_html=True)

# Regroupement des investissements par catégorie pour une meilleure organisation
investissements_categories = {
    "Par terrain (multiplié par le nombre de terrains)": [
        "Avance de terrain", "Construction", "Gazon", "Caméras de surveillance", "Caméras de filmage"
    ],
    "Communs (non multiplié)": [
        "Publicités", "Stock initial", "Social Media et App", "Divers", "Création d'association"
    ]
}

with st.form(key="investissements_form"):
    for categorie, items in investissements_categories.items():
        st.markdown(f"#### {categorie}")
        
        # Utiliser des colonnes pour organiser les champs
        cols = st.columns(2)
        half = len(items) // 2 + len(items) % 2
        
        for i, item in enumerate(items[:half]):
            with cols[0]:
                montant = st.number_input(
                    f"{item}",
                    min_value=0.0,
                    value=st.session_state.charges_investissement.get(item, 0.0),
                    step=1000.0,
                    format="%.2f",
                    key=f"inv_{categorie}_{i}"
                )
                st.session_state.charges_investissement[item] = montant
        
        for i, item in enumerate(items[half:]):
            with cols[1]:
                montant = st.number_input(
                    f"{item}",
                    min_value=0.0,
                    value=st.session_state.charges_investissement.get(item, 0.0),
                    step=1000.0,
                    format="%.2f",
                    key=f"inv_{categorie}_{i + half}"
                )
                st.session_state.charges_investissement[item] = montant
        
        st.markdown("---")
    
    # Bouton pour soumettre les modifications
    inv_submitted = st.form_submit_button("Mettre à jour les investissements")
    if inv_submitted:
        st.success("Investissements mis à jour! Les calculs ont été recalculés.")
        indicateurs = calculer_indicateurs()  # Recalculer les indicateurs

# Afficher le tableau des investissements
inv_data = []
for categorie, items in investissements_categories.items():
    subtotal = 0
    for item in items:
        montant = st.session_state.charges_investissement.get(item, 0.0)
        subtotal += montant
        inv_data.append({
            "Catégorie": categorie,
            "Investissement": item,
            "Montant (DH)": f"{montant:.2f} DH"
        })
    
    # Ajouter un sous-total pour chaque catégorie
    if categorie == "Par terrain (multiplié par le nombre de terrains)":
        montant_total = subtotal * st.session_state.nb_terrains
        inv_data.append({
            "Catégorie": categorie,
            "Investissement": f"Sous-total ({st.session_state.nb_terrains} terrains)",
            "Montant (DH)": f"{montant_total:.2f} DH"
        })
    else:
        inv_data.append({
            "Catégorie": categorie,
            "Investissement": "Sous-total",
            "Montant (DH)": f"{subtotal:.2f} DH"
        })

# Ajouter une ligne de total pour les investissements
inv_data.append({
    "Catégorie": "",
    "Investissement": "📊 TOTAL INVESTISSEMENT",
    "Montant (DH)": f"{indicateurs['total_investissement']:.2f} DH"
})

df_inv = pd.DataFrame(inv_data)
st.dataframe(df_inv, use_container_width=True)

# 7. Graphiques en camembert pour la répartition des coûts
st.markdown('<p class="sub-header">📉 Répartition des coûts</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Camembert des coûts variables par service
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    labels_services = [f"{st.session_state.services[service]} {service}" for service in st.session_state.services]
    valeurs = [indicateurs['couts_services'][service] for service in st.session_state.services]
    
    # Filtrer les services sans coûts pour une meilleure lisibilité
    filtered_labels = []
    filtered_values = []
    for label, value in zip(labels_services, valeurs):
        if value > 0:
            filtered_labels.append(label)
            filtered_values.append(value)
    
    if sum(filtered_values) > 0:
        ax1.pie(filtered_values, labels=filtered_labels, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        plt.title('Répartition des coûts variables par service')
        st.pyplot(fig1)
    else:
        st.warning("Aucun coût variable à afficher. Veuillez définir des services avec des coûts.")

with col2:
    # Camembert des charges fixes
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    labels_charges = [f"{charges_emojis.get(charge, '📝')} {charge}" for charge in st.session_state.charges_mensuelles]
    valeurs_charges = [st.session_state.charges_mensuelles[charge] for charge in st.session_state.charges_mensuelles]
    
    # Filtrer les charges sans montants pour une meilleure lisibilité
    filtered_labels_charges = []
    filtered_values_charges = []
    for label, value in zip(labels_charges, valeurs_charges):
        if value > 0:
            filtered_labels_charges.append(label)
            filtered_values_charges.append(value)
    
    if sum(filtered_values_charges) > 0:
        ax2.pie(filtered_values_charges, labels=filtered_labels_charges, autopct='%1.1f%%', startangle=90)
        ax2.axis('equal')
        plt.title('Répartition des charges fixes mensuelles')
        st.pyplot(fig2)
    else:
        st.warning("Aucune charge fixe à afficher. Veuillez définir des charges avec des montants.")

# 8. Analyse de rentabilité
st.markdown('<p class="sub-header">📈 Analyse de rentabilité</p>', unsafe_allow_html=True)

col_rentab1, col_rentab2 = st.columns(2)

with col_rentab1:
    st.markdown("### 🏁 Seuil de rentabilité")
    st.metric(
        label="Seuil de rentabilité mensuel",
        value=f"{indicateurs['seuil_rentabilite']:.2f} DH"
    )
    st.write(f"Vous devez générer au moins **{indicateurs['seuil_rentabilite']:.2f} DH** de revenus mensuels pour couvrir tous vos coûts.")
    
    # Progression vers le seuil de rentabilité
    if indicateurs['revenu_brut'] > 0:
        progress_value = min(indicateurs['revenu_brut'] / indicateurs['seuil_rentabilite'], 1.0)
        st.progress(progress_value)
        if progress_value >= 1.0:
            st.success(f"✅ Vous avez dépassé le seuil de rentabilité de {(progress_value - 1.0) * 100:.1f}%!")
        else:
            st.warning(f"⚠️ Vous êtes à {progress_value * 100:.1f}% du seuil de rentabilité.")
    else:
        st.warning("⚠️ Aucun revenu n'est généré actuellement. Définissez des services actifs.")

with col_rentab2:
    st.markdown("### ⏱️ Retour sur investissement")
    
    if indicateurs['profit_net'] > 0:
        st.metric(
            label="Temps estimé pour rentabiliser l'investissement",
            value=f"{indicateurs['temps_retour']:.1f} mois"
        )
        annees = int(indicateurs['temps_retour'] // 12)
        mois_restants = int(indicateurs['temps_retour'] % 12)
        if annees > 0:
            st.write(f"Votre investissement total de **{indicateurs['total_investissement']:.2f} DH** sera rentabilisé en **{annees} ans et {mois_restants} mois**.")
        else:
            st.write(f"Votre investissement total de **{indicateurs['total_investissement']:.2f} DH** sera rentabilisé en **{mois_restants} mois**.")
    else:
        st.error("❌ Avec les paramètres actuels, votre activité n'est pas rentable. Ajustez vos paramètres pour générer un profit net positif.")

# 9. Comparaison avec différents nombres de terrains
st.markdown('<p class="sub-header">🔄 Comparaison du nombre de terrains</p>', unsafe_allow_html=True)

# Calculer les indicateurs pour différents nombres de terrains
max_terrains = 5
comparaison_terrains = []

for n_terrains in range(1, max_terrains + 1):
    # Sauvegarde temporaire du nombre actuel de terrains
    nombre_terrains_actuel = st.session_state.nb_terrains
    
    # Définir temporairement le nombre de terrains pour le calcul
    st.session_state.nb_terrains = n_terrains
    comp_indicateurs = calculer_indicateurs()
    
    # Restaurer le nombre de terrains d'origine
    st.session_state.nb_terrains = nombre_terrains_actuel
    
    comparaison_terrains.append({
        "Nombre de terrains": n_terrains,
        "Revenu mensuel": comp_indicateurs['revenu_brut'],
        "Coût total": comp_indicateurs['cout_total'],
        "Profit net": comp_indicateurs['profit_net'],
        "Profit par associé": comp_indicateurs['profit_par_associe'],
        "Investissement initial": comp_indicateurs['total_investissement'],
        "ROI mensuel": comp_indicateurs['roi_mensuel'],
        "Temps de retour (mois)": comp_indicateurs['temps_retour'] if comp_indicateurs['profit_net'] > 0 else float('inf')
    })

df_comparaison = pd.DataFrame(comparaison_terrains)

# Formater les colonnes pour un meilleur affichage
df_comparaison["Revenu mensuel"] = df_comparaison["Revenu mensuel"].apply(lambda x: f"{x:.2f} DH")
df_comparaison["Coût total"] = df_comparaison["Coût total"].apply(lambda x: f"{x:.2f} DH")
df_comparaison["Profit net"] = df_comparaison["Profit net"].apply(lambda x: f"{x:.2f} DH")
df_comparaison["Profit par associé"] = df_comparaison["Profit par associé"].apply(lambda x: f"{x:.2f} DH")
df_comparaison["Investissement initial"] = df_comparaison["Investissement initial"].apply(lambda x: f"{x:.2f} DH")
df_comparaison["ROI mensuel"] = df_comparaison["ROI mensuel"].apply(lambda x: f"{x:.2f}%")
df_comparaison["Temps de retour (mois)"] = df_comparaison["Temps de retour (mois)"].apply(lambda x: f"{x:.1f}" if x != float('inf') else "N/A")

st.dataframe(df_comparaison, use_container_width=True)

# Graphique de comparaison des profits nets selon le nombre de terrains
fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
x = range(1, max_terrains + 1)
y = [comp['Profit net'] for comp in comparaison_terrains]
y_clean = [float(val.split(' ')[0]) for val in y]  # Extraire les valeurs numériques

ax_comp.bar(x, y_clean, color='#2196F3')
ax_comp.set_xlabel('Nombre de terrains')
ax_comp.set_ylabel('Profit net mensuel (DH)')
ax_comp.set_title('Évolution du profit net en fonction du nombre de terrains')
ax_comp.set_xticks(x)
ax_comp.set_xticklabels([f'{i} terrain{"s" if i>1 else ""}' for i in x])

# Ajouter les valeurs au-dessus des barres
for i, v in enumerate(y_clean):
    ax_comp.text(i + 1, v + 1000, f'{v:.2f} DH', ha='center')

st.pyplot(fig_comp)

# 10. Conclusion et recommandations
st.markdown('<p class="sub-header">🎯 Conclusion et Recommandations</p>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    if indicateurs['profit_net'] > 0:
        st.success(f"✅ Avec les paramètres actuels, votre projet de mini football avec {st.session_state.nb_terrains} terrain{'s' if st.session_state.nb_terrains > 1 else ''} est rentable.")
        
        # Recommandations
        st.markdown("### 💡 Recommandations")
        
        # Trouver le service le plus rentable
        service_rentable = max(st.session_state.services, key=lambda s: indicateurs['marges_services'][s])
        emoji_rentable = st.session_state.services[service_rentable]
        
        st.markdown(f"""
        1. **Service le plus rentable**: Le {emoji_rentable} {service_rentable} est votre service le plus rentable. Envisagez de concentrer vos efforts marketing sur ce service.
        
        2. **Nombre de terrains optimal**: D'après notre simulation, {df_comparaison['Profit net'].tolist().index(max(df_comparaison['Profit net'].tolist())) + 1} terrain(s) semble être le nombre optimal pour maximiser votre rentabilité par rapport à l'investissement.
        
        3. **Temps de retour sur investissement**: Avec {st.session_state.nb_terrains} terrain{'s' if st.session_state.nb_terrains > 1 else ''}, vous pourriez rentabiliser votre investissement en {indicateurs['temps_retour']:.1f} mois.
        """)
    else:
        st.error(f"❌ Avec les paramètres actuels, votre projet de mini football avec {st.session_state.nb_terrains} terrain{'s' if st.session_state.nb_terrains > 1 else ''} n'est pas rentable.")
        
        # Recommandations pour améliorer la rentabilité
        st.markdown("### 💡 Recommandations pour améliorer la rentabilité")
        
        # Identifier le service avec la meilleure marge unitaire
        meilleure_marge_service = max(st.session_state.services, key=lambda s: (st.session_state.prix_vente[s] - st.session_state.cout_unitaire[s]))
        emoji_meilleure_marge = st.session_state.services[meilleure_marge_service]
        
        st.markdown(f"""
        1. **Augmenter les prix**: Envisagez d'augmenter légèrement les prix de vos services, en particulier pour les services à forte demande.
        
        2. **Optimiser les coûts**: Identifiez les postes de dépenses les plus élevés et cherchez des moyens de les réduire sans compromettre la qualité.
        
        3. **Développer les services rentables**: Le {emoji_meilleure_marge} {meilleure_marge_service} a la meilleure marge unitaire. Concentrez vos efforts pour augmenter le nombre de commandes pour ce service.
        
        4. **Revoir le nombre de terrains**: Selon notre simulation, {df_comparaison['Profit net'].tolist().index(max(df_comparaison['Profit net'].tolist())) + 1} terrain(s) pourrait être plus optimal pour votre situation actuelle.
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("### ⚽ SimuProfit - Business Plan Mini Football")
st.markdown("Une application pour simuler et optimiser la rentabilité de vos terrains de mini football.")
