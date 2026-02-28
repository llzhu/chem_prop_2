## ChemProp Implimentation with Streamlit 

- [ ] [ChemProp is a message passing neural network for molecular property prediction:](https://github.com/chemprop/chemprop)
Yang, et al, **Analyzing Learned Molecular Representations for Property Prediction**, _J. Chem. Inf. Model._ 2019, 59, 3370−3388

### Delaney Solubility Data Set:
Delaney, **Estimating Aqueous Solubility Directly from Molecular Structure**, _J. Chem. Inf. Comput. Sci._ 2004, 44, 3, 1000–1005

### Thrombin_IC50 is the IC50 of inhibitory activity against human thrombin CHEMBL204 with the following query:
activities = activity.filter(target_chembl_id="CHEMBL204").filter(standard_type='IC50').filter(standard_relation='=')
            .only(['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'standard_units'])

