adfr_prepare = "/home/jyzha/software/adfr/bin/prepare_receptor"
adfr_autosite = "/home/jyzha/software/adfr/bin/autosite"

casc_sel = '(resname ALA and (name CA)) or (resname GLY and (name CA)) or (resname ARG and (name CA or name CG or name NE)) or (resname ASN and (name CA or name CG))\
 or (resname ASP and (name CA or name CG)) or (resname CYS and (name CA or name SG)) or (resname GLN and (name CA or name CG or name OE1)) or (resname GLU and (name CA or name CG or name OE1)) \
 or (resname HIS and (name CA or name CG)) or (resname ILE and (name CA or name CG1)) or (resname LEU and (name CA or name CG)) or (resname LYS and (name CA or name CG or name CE))\
 or (resname MET and (name CA or name CG or name CE)) or (resname PHE and (name CA or name CG)) or (resname PRO and (name CA or name CG)) or (resname SER and (name CA or name OG)) \
 or (resname THR and (name CA or name OG1)) or (resname TRP and (name CA or name CG)) or (resname TYR and (name CA or name CG)) or (resname VAL and (name CA or name CG1))'

aa_chi_dict = { 
    'ALA': [],
    'GLY': [],
    # 精氨酸（通常保持质子化）
    'ARG': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
    
    # 天冬氨酸及其质子化变体
    'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'ASH': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],  # 质子化ASP
    
    # 半胱氨酸及其去质子化变体
    'CYS': [['N', 'CA', 'CB', 'SG']],
    'CYM': [['N', 'CA', 'CB', 'SG']],  # 去质子化CYS
    
    # 谷氨酰胺及其质子化变体
    'GLN': [['N', 'CA', 'CB', 'CG'],
            ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLH': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],  # 质子化GLU
    
    # 组氨酸不同质子化状态（已由您补充）
    'HIS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
    'HIE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
    'HID': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
    'HIP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
    
    # 赖氨酸及其去质子化变体
    'LYS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
    'LYN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],  # 去质子化LYS
    
    # 其他标准氨基酸
    'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD']],
    'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'MET': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'],
            ['CB', 'CG', 'SD', 'CE']],
    'PHE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'PRO': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']],
    'SER': [['N', 'CA', 'CB', 'OG']],
    'THR': [['N', 'CA', 'CB', 'OG1']],
    'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    
    # 酪氨酸及其去质子化变体（罕见）
    'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'TYM': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],  # 去质子化TYR
    
    'VAL': [['N', 'CA', 'CB', 'CG1']]
}