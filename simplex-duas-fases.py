import re

EPS = 1e-9

class LinearProgram:
    def __init__(self):
        self.sense = None               # 'MAX' ou 'MIN'
        self.objective = {}             # var -> coeff
        self.constraints = []           # Dicionario
        self.var_types = {}             # '>=0' | '<=0' | 'livre'
        self.variables = []             # Variaveis originais

        # Transformada
        self.trans_vars = []          
        self.orig_to_trans = {}        

        # Forma padrão
        self.A = []       
        self.b = []
        self.relations = []
        self.all_var_names = []       
        self.var_index = {}
        self.init_basis = []       
        self.artificial_cols = []   

        # Solução
        self.solution = {}
        self.opt_value = None
        self.status = 'Not solved'
        self.phase1_tableau_history = []
        self.phase2_tableau_history = []

    def parse_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            raw_lines = [ln.rstrip() for ln in f]
        lines = [ln.strip() for ln in raw_lines if ln.strip() != '']
        if not lines:
            raise ValueError("Arquivo vazio")
        # Objetivo
        self._parse_objective(lines[0])
        # Encontra o dominio de início
        domain_start = len(lines)
        for i in range(1, len(lines)):
            ln = lines[i]
            if re.match(r'^[A-Za-z]\w*\s*(?:<=|>=|\blivre\b)', ln):
                domain_start = i
                break

        for i in range(1, domain_start):
            self._parse_constraint(lines[i])
        for i in range(domain_start, len(lines)):
            self._parse_variable_domain(lines[i])
        for v in self.variables:
            if v not in self.var_types:
                self.var_types[v] = '>=0'

    def _parse_objective(self, line):
        parts = line.strip().split(None, 1)
        if not parts:
            raise ValueError("Linha de objetivo inválida")
        self.sense = parts[0].upper()
        if self.sense not in ('MAX', 'MIN'):
            raise ValueError("Sentido da função objetivo deve ser MAX ou MIN")
        expr = parts[1] if len(parts) > 1 else ''
        expr = expr.replace('-', '+-')
        terms = [t.strip() for t in expr.split('+') if t.strip()]
        for term in terms:
            m = re.match(r'^([+-]?\s*(?:\d+(?:\.\d*)?|\.\d+)?)(?:\s*\*?\s*)?([A-Za-z]\w*)$', term)
            if not m:
                raise ValueError("Termo de objetivo inválido: '{}'".format(term))
            coeff_s, var = m.groups()
            coeff_s = coeff_s.replace(' ', '')
            if coeff_s in ('', '+', None):
                coeff = 1.0
            elif coeff_s == '-':
                coeff = -1.0
            else:
                coeff = float(coeff_s)
            self.objective[var] = coeff
            if var not in self.variables:
                self.variables.append(var)

    def _parse_constraint(self, line):
        # encontra operador
        m = re.search(r'(<=|>=|=)', line)
        if not m:
            raise ValueError("Sem operador relacional na restrição: " + line)

        op = m.group(1)
        lhs = line[:m.start()].strip()
        rhs = float(line[m.end():].strip())

        # normalizar sinais: troca "-" por "+-" e divide por "+"
        lhs_norm = lhs.replace('-', '+-')
        terms = [t.strip() for t in lhs_norm.split('+') if t.strip()]
        coeffs = {}

        for term in terms:
            # captura formatos válidos:
            m2 = re.match(r'^([+-]?\s*(?:\d+(?:\.\d*)?)?)\s*\*?\s*([A-Za-z]\w*)$', term)

            if not m2:
                raise ValueError("Termo inválido na restrição: '{}' (linha: {})".format(term, line))

            coeff_s, var = m2.groups()
            coeff_s = coeff_s.replace(" ", "")

            if coeff_s in ("", "+", None):
                coeff = 1.0
            elif coeff_s == "-":
                coeff = -1.0
            else:
                coeff = float(coeff_s)

            coeffs[var] = coeffs.get(var, 0.0) + coeff

            if var not in self.variables:
                self.variables.append(var)

        self.constraints.append({
            "coeffs": coeffs,
            "rel": op,
            "rhs": rhs
        })

    def _parse_variable_domain(self, line):
        line = line.strip()
        m_free = re.match(r'^([A-Za-z]\w*)\s+livre$', line)
        if m_free:
            var = m_free.group(1)
            self.var_types[var] = 'free'
            if var not in self.variables:
                self.variables.append(var)
            return
        m2 = re.match(r'^([A-Za-z]\w*)\s*(<=|>=)\s*([+-]?\d+(?:\.\d*)?|\.\d+)$', line)
        if not m2:
            return
        var, op, val_s = m2.groups()
        val = float(val_s)
        if op == '>=' and abs(val - 0.0) < EPS:
            self.var_types[var] = '>=0'
        elif op == '<=' and abs(val - 0.0) < EPS:
            self.var_types[var] = '<=0'
        else:
            self.var_types[var] = op + str(val)
        if var not in self.variables:
            self.variables.append(var)

    def transform_variables(self):
        self.trans_vars = []
        self.orig_to_trans = {}
        for var in self.variables:
            vtype = self.var_types.get(var, '>=0')
            if vtype == '>=0':
                t = var
                self.trans_vars.append(t)
                self.orig_to_trans[var] = [(t, 1.0)]
            elif vtype == '<=0':
                t = var + "_neg"
                self.trans_vars.append(t)
                self.orig_to_trans[var] = [(t, -1.0)]
            elif vtype == 'free':
                p = var + "_pos"; n = var + "_neg"
                self.trans_vars.append(p); self.trans_vars.append(n)
                self.orig_to_trans[var] = [(p, 1.0), (n, -1.0)]
            else:
                t = var
                self.trans_vars.append(t)
                self.orig_to_trans[var] = [(t, 1.0)]

    def build_standard_form(self):
        self.transform_variables()
        slack_names = []
        surplus_names = []
        artificial_names = []

        # Cada restrição contribui com extras
        for i, constr in enumerate(self.constraints):
            rel = constr["rel"]
            if rel == "<=":
                slack_names.append(f"s_{i}")
            elif rel == ">=":
                surplus_names.append(f"e_{i}")
                artificial_names.append(f"a_{i}")
            elif rel == "=":
                artificial_names.append(f"a_{i}")

            else:
                raise ValueError("Operador relacional desconhecido: " + rel)

        # Nome final das colunas extras (ordem fixa e global)
        extra_names = slack_names + surplus_names + artificial_names
        self.all_var_names = list(self.trans_vars) + list(extra_names)
        self.var_index = {name: j for j, name in enumerate(self.all_var_names)}

        n = len(self.all_var_names)    # número total de colunas
        m = len(self.constraints)      # número de restrições
        A = []
        b = []
        init_basis = [None] * m
        self.artificial_cols = []

        for i, constr in enumerate(self.constraints):
            row = [0.0] * n

            for orig_var, coeff in constr["coeffs"].items():
                for tname, sign in self.orig_to_trans[orig_var]:
                    j = self.var_index[tname]
                    row[j] += coeff * sign

            rel = constr["rel"]

            if rel == "<=":
                # slack +1
                sname = f"s_{i}"
                j = self.var_index[sname]
                row[j] = 1.0
                init_basis[i] = sname

            elif rel == ">=":
                ename = f"e_{i}"
                j = self.var_index[ename]
                row[j] = -1.0
                aname = f"a_{i}"
                j = self.var_index[aname]
                row[j] = 1.0
                init_basis[i] = aname
                self.artificial_cols.append(j)

            elif rel == "=":
                aname = f"a_{i}"
                j = self.var_index[aname]
                row[j] = 1.0
                init_basis[i] = aname
                self.artificial_cols.append(j)

            else:
                raise ValueError("Operador relacional inválido: " + rel)

            A.append(row)
            b.append(constr["rhs"])

        # Salvar internamente
        self.A = A
        self.b = b
        self.init_basis = init_basis
        self.relations = [c["rel"] for c in self.constraints]

    def build_tableau(self, obj_coeffs, maximize=True):
        m = len(self.A)
        n = len(self.all_var_names)
        tableau = []
        for i in range(m):
            row = [0.0] * (n + 1)
            for j in range(n):
                row[j] = self.A[i][j]
            row[-1] = self.b[i]
            tableau.append(row)

        obj_row = [0.0] * (n + 1)
        for name, coeff in obj_coeffs.items():
            if name in self.var_index:
                j = self.var_index[name]
                obj_row[j] = coeff if maximize else -coeff
        obj_row[-1] = 0.0  # valor inicial da FO
        tableau.append(obj_row)
        return tableau

    def _choose_entering(self, obj_row):
        for j, val in enumerate(obj_row[:-1]):
            if val > EPS:  # Apenas se for estritamente positivo
                return j
        return None  # Nenhum candidato positivo encontrado -> Ótimo atingido

    def _choose_leaving(self, tableau, entering_col):
        best_ratio = float('inf')
        best_i = None
        m = len(tableau) - 1
        
        for i in range(m):
            a_ij = tableau[i][entering_col]
            if a_ij <= EPS:
                continue
            b_i = tableau[i][-1]
            # valor negativo no RHS
            if b_i < -EPS:  
                continue
            ratio = b_i / a_ij
            if ratio >= -EPS and ratio < best_ratio - EPS:
                best_ratio = ratio
                best_i = i
        return best_i

    def _pivot(self, tableau, pivot_row, pivot_col):
        pv = tableau[pivot_row][pivot_col]
        if abs(pv) < EPS:
            return False  

        tableau[pivot_row] = [x / pv for x in tableau[pivot_row]]
        
        for i in range(len(tableau)):
            if i == pivot_row:
                continue
            factor = tableau[i][pivot_col]
            if abs(factor) > EPS:
                tableau[i] = [tableau[i][j] - factor * tableau[pivot_row][j] 
                            for j in range(len(tableau[0]))]
        return True

    def run_simplex(self, tableau, basis_vars, phase=1):
        max_iters = 5000  
        it = 0
        
        # Armazena o tableau inicial
        status_msg = "Inicial"
        if phase == 1:
            self.phase1_tableau_history.append(self._format_tableau(tableau, basis_vars, status_msg))
        else:
            self.phase2_tableau_history.append(self._format_tableau(tableau, basis_vars, status_msg))

        while it < max_iters:
            it += 1
            obj_row = tableau[-1]
            
            entering = self._choose_entering(obj_row)
            if entering is None:
                # Solução ótima encontrada (todos coeficientes <= 0)
                status_str = "Ótimo"
                if phase == 1:
                    self.phase1_tableau_history.append(self._format_tableau(tableau, basis_vars, status_str))
                else:
                    self.phase2_tableau_history.append(self._format_tableau(tableau, basis_vars, status_str))
                return tableau[-1][-1], tableau, 'optimal'
                
            leaving = self._choose_leaving(tableau, entering)
            if leaving is None:
                status_str = "Ilimitado"
                if phase == 1:
                    self.phase1_tableau_history.append(self._format_tableau(tableau, basis_vars, status_str))
                else:
                    self.phase2_tableau_history.append(self._format_tableau(tableau, basis_vars, status_str))
                return None, tableau, 'unbounded'

            old_basis_var = basis_vars[leaving] if leaving < len(basis_vars) else "?"
            status_str = f"Iteração {it} (Entra: {self.all_var_names[entering]}, Sai: {old_basis_var})"

            if phase == 1:
                self.phase1_tableau_history.append(self._format_tableau(tableau, basis_vars, status_str))
            else:
                self.phase2_tableau_history.append(self._format_tableau(tableau, basis_vars, status_str))
            
            # Pivot
            if not self._pivot(tableau, leaving, entering):
                status_str = "Pivot falhou (Numerico)"
                if phase == 1:
                    self.phase1_tableau_history.append(self._format_tableau(tableau, basis_vars, status_str))
                return None, tableau, 'pivot_failed'

            if leaving < len(basis_vars):
                basis_vars[leaving] = self.all_var_names[entering]
            
        status_str = f"Máximo de iterações ({max_iters})"
        if phase == 1:
            self.phase1_tableau_history.append(self._format_tableau(tableau, basis_vars, status_str))
        else:
            self.phase2_tableau_history.append(self._format_tableau(tableau, basis_vars, status_str))
        return None, tableau, 'max_iterations'

    def _format_tableau(self, tableau, basis_vars, status):
        formatted = f"{status}:\n"
        
        # Header
        headers = self.all_var_names + ['RHS']
        header_line = "Básica | " + " | ".join(f"{h:>10}" for h in headers) + " |"
        formatted += header_line + "\n"
        formatted += "-" * len(header_line) + "\n"
        
        # Linhas
        for i in range(len(tableau) - 1):
            basis_var = basis_vars[i] if i < len(basis_vars) else "?"
            row_str = f"{basis_var:>6} | " + " | ".join(f"{x:>10.4f}" for x in tableau[i]) + " |"
            formatted += row_str + "\n"
        
        # Linha objetivo
        formatted += "FO     | " + " | ".join(f"{x:>10.4f}" for x in tableau[-1]) + " |\n"
        
        return formatted

    def phase_one(self):
        if not self.artificial_cols:
            tableau = self.build_tableau({}, maximize=True)
            basis = list(self.init_basis)
            return 0.0, 'optimal', tableau, basis

        obj_coeffs = {name: 0.0 for name in self.all_var_names}
        for idx in self.artificial_cols:
            name = self.all_var_names[idx]
            obj_coeffs[name] = -1.0  

        tableau = self.build_tableau(obj_coeffs, maximize=True)
        
        # Arruma a linha objetiva
        obj_row = tableau[-1]
        for i, basis_var in enumerate(self.init_basis):
            if basis_var and basis_var.startswith('a_'):
                row_i = tableau[i]
                coeff_in_obj = obj_coeffs.get(basis_var, 0.0)
                if abs(coeff_in_obj) > EPS:
                    tableau[-1] = [obj_row[j] - coeff_in_obj * row_i[j] 
                                for j in range(len(obj_row))]
                    obj_row = tableau[-1]

        basis = list(self.init_basis)
        val, final_tab, status = self.run_simplex(tableau, basis, phase=1)
        
        return val, status, final_tab, basis

    def phase_two(self, phase1_tableau, phase1_basis):
        if self.artificial_cols and abs(self.phase1_value) > EPS:
            self.status = 'Infeasible'
            return None, 'infeasible'

        # Remove colunas artificiais existentes
        if self.artificial_cols:
            keep_cols = [j for j in range(len(self.all_var_names)) 
                        if not self.all_var_names[j].startswith('a_')]
            new_all_var_names = [self.all_var_names[j] for j in keep_cols]
            
            # Constroi outro tableau
            new_tab = []
            for i in range(len(phase1_tableau)):
                new_row = [phase1_tableau[i][j] for j in keep_cols] + [phase1_tableau[i][-1]]
                new_tab.append(new_row)
                
            self.all_var_names = new_all_var_names
            self.var_index = {name: idx for idx, name in enumerate(self.all_var_names)}
        else:
            new_tab = [row[:] for row in phase1_tableau]

        # Remove as variaveis artificiais restantes
        basis = []
        for i, basis_var in enumerate(phase1_basis):
            if basis_var and not basis_var.startswith('a_'):
                basis.append(basis_var)
            else:
                # Encontra uma variavel não artificial  
                found = False
                for j in range(len(self.all_var_names)):
                    if abs(new_tab[i][j]) > EPS:
                        self._pivot(new_tab, i, j)
                        basis.append(self.all_var_names[j])
                        found = True
                        break
                if not found:
                    continue

        obj_coeffs = {name: 0.0 for name in self.all_var_names}
        for orig_var, coeff in self.objective.items():
            transforms = self.orig_to_trans.get(orig_var, [])
            for tname, sign in transforms:
                if tname in obj_coeffs:
                    obj_coeffs[tname] += coeff * sign

        maximize = (self.sense == 'MAX')
        
        # Constroi a linha objetiva
        obj_row = [0.0] * (len(self.all_var_names) + 1)
        for name, coeff in obj_coeffs.items():
            if name in self.var_index:
                j = self.var_index[name]
                obj_row[j] = coeff if maximize else -coeff

        # Subtrai a variavel básica
        for i, basis_var in enumerate(basis):
            if basis_var and basis_var in obj_coeffs:
                coeff = obj_coeffs[basis_var]
                multiplier = coeff if maximize else -coeff
                row_i = new_tab[i]
                obj_row = [obj_row[j] - multiplier * row_i[j] 
                        for j in range(len(obj_row))]

        new_tab[-1] = obj_row

        # Executa a fase 2 do simplex
        val, final_tab, status = self.run_simplex(new_tab, basis, phase=2)
        
        if status == 'optimal':
            # Extrai a solução
            var_vals = {name: 0.0 for name in self.all_var_names}
            for i, basis_var in enumerate(basis):
                if basis_var and i < len(final_tab) - 1:
                    var_vals[basis_var] = final_tab[i][-1]
            
            self._extract_original_solution(var_vals)
            
            # Calcula o valor otimo
            obj_val = 0.0
            for var, coeff in self.objective.items():
                obj_val += coeff * self.solution.get(var, 0.0)
            self.opt_value = obj_val
            
            return val, 'optimal'
        else:
            return None, status

    def _extract_original_solution(self, var_vals):
        sol = {}
        for orig_var in self.variables:
            val = 0.0
            for tname, sign in self.orig_to_trans[orig_var]:
                val += sign * var_vals.get(tname, 0.0)
            sol[orig_var] = val
        self.solution = sol

    def solve(self):
        try:
            self.phase1_tableau_history = []
            self.phase2_tableau_history = []
            
            self.build_standard_form()
            
            print(f"Problema transformado: {len(self.all_var_names)} variáveis, {len(self.constraints)} restrições")
            print(f"Variáveis artificiais: {len(self.artificial_cols)}")
            
            # FASE 1
            p1_val, p1_status, p1_tab, p1_basis = self.phase_one()
            self.phase1_value = p1_val
            
            print(f"Fase 1: status={p1_status}, valor={p1_val}")
            
            if p1_status != 'optimal':
                self.status = f'Phase1_{p1_status}'
                return
                
            if self.artificial_cols and abs(p1_val) > EPS:
                self.status = 'Infeasible'
                return

            # FASE 2
            val, status = self.phase_two(p1_tab, p1_basis)
            self.status = status
            
            print(f"Fase 2: status={status}")
            
        except Exception as e:
            self.status = f'Error: {str(e)}'
            import traceback
            print(f"Erro durante solução: {e}")
            traceback.print_exc()

    def calculate_objective_value(self):
        if not self.solution:
            return 0.0
        val = 0.0
        for v, coeff in self.objective.items():
            val += coeff * self.solution.get(v, 0.0)
        return val

    def verify_constraints(self):
        results = []
        for i, constr in enumerate(self.constraints, 1):
            lhs = 0.0
            for var, coeff in constr['coeffs'].items():
                lhs += coeff * self.solution.get(var, 0.0)
            rel = constr['rel']
            rhs = constr['rhs']
            
            if rel == '<=':
                satisfied = lhs <= rhs + EPS
                results.append(f"R{i}: {lhs:.6f} <= {rhs:.6f} {'✓' if satisfied else '✗'}")
            elif rel == '>=':
                satisfied = lhs >= rhs - EPS
                results.append(f"R{i}: {lhs:.6f} >= {rhs:.6f} {'✓' if satisfied else '✗'}")
            else:  # '='
                satisfied = abs(lhs - rhs) < EPS
                results.append(f"R{i}: {lhs:.6f} = {rhs:.6f} {'✓' if satisfied else '✗'}")
                
        return results

    def generate_results_file(self, output_filename="/home/beatriznahas/BCC/Semestre_5-6/Otimização/simplex/testeFinal-BeatrizN_JoaoPlinta_RafaelAlves.txt"):
        self.solve()
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("Solução do Método Simplex de Duas Fases\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Status: {self.status}\n")
            f.write(f"Valor da FO: {self.calculate_objective_value():.6f}\n\n")
            
            if self.status == 'optimal':
                f.write("Variáveis:\n")
                for var in self.variables:
                    val = self.solution.get(var, 0.0)
                    f.write(f"  {var} = {val:.6f}\n")
                
                f.write("\nVerificação das Restrições:\n")
                for line in self.verify_constraints():
                    f.write(f"  {line}\n")
            else:
                f.write("Solução não ótima encontrada.\n")
            
            # Adicionar tableaus da Fase 1
            if self.phase1_tableau_history:
                f.write("\n\n" + "="*60 + "\n")
                f.write("FASE 1 - TABLEAUS\n")
                f.write("="*60 + "\n")
                for i, tableau_str in enumerate(self.phase1_tableau_history):
                    f.write(f"\n{tableau_str}\n")
            
            # Adicionar tableaus da Fase 2
            if self.phase2_tableau_history:
                f.write("\n\n" + "="*60 + "\n")
                f.write("FASE 2 - TABLEAUS\n")
                f.write("="*60 + "\n")
                for i, tableau_str in enumerate(self.phase2_tableau_history):
                    f.write(f"\n{tableau_str}\n")

        print(f"Arquivo gerado: {output_filename}")

# FUNÇÃO: Executa o algoritmo
def main():
    import sys
    filename = "/home/beatriznahas/BCC/Semestre_5-6/Otimização/simplex/TesteFinal_lp.txt"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    lp = LinearProgram()
    try:
        lp.parse_file(filename)
        print("Arquivo lido com sucesso!")
        print(f"Variáveis: {lp.variables}")
        print(f"Tipo de objetivo: {lp.sense}")
        print(f"Número de restrições: {len(lp.constraints)}")
        
        lp.generate_results_file("/home/beatriznahas/BCC/Semestre_5-6/Otimização/simplex/testeFinal-BeatrizN_JoaoPlinta_RafaelAlves.txt")
        
        print(f"\nStatus final: {lp.status}")
        print(f"Valor ótimo: {lp.calculate_objective_value():.6f}")
        
        if lp.status == 'optimal':
            print("\nSolução ótima:")
            for var in lp.variables:
                print(f"  {var} = {lp.solution.get(var, 0.0):.6f}")
                
    except Exception as e:
        print(f"Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()