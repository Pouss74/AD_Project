import streamlit as st

## ICI il y aura la Web app qui appelle les différentes fonctions et trucs à afficher de backend.py
# Supposons que votre script converti s'appelle notebook_script.py
import backend

# Titre de l'application
st.title('Application Web utilisant Streamlit et un Notebook Jupyter')

# Vous pouvez maintenant appeler les fonctions du script converti
result = backend.ma_fonction()

# Afficher le résultat dans Streamlit
st.write(result)

# Vous pouvez également afficher d'autres types de contenu comme des graphiques
if hasattr(notebook_script, 'mon_graphe'):
    st.pyplot(notebook_script.mon_graphe())
