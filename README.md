# Método Simplex de Duas Fases

Implementação do algoritmo Simplex de duas fases para resolução de problemas de programação linear.

## Autores

- Beatriz Regina Nahas
- João Pedro dos Santos Henrique Plinta
- Rafael Correa Alves

## Descrição

Este projeto implementa o método Simplex de duas fases, capaz de resolver problemas de programação linear (PL) com diferentes tipos de restrições e variáveis. O algoritmo:

- Suporta problemas de maximização e minimização
- Trabalha com restrições do tipo `<=`, `>=` e `=`
- Aceita variáveis livres, não-negativas e não-positivas
- Gera histórico completo dos tableaus em cada fase
- Verifica automaticamente a satisfação das restrições

## Funcionalidades

### Fase 1
- Introduz variáveis artificiais para obter uma solução básica inicial viável
- Minimiza a soma das variáveis artificiais
- Verifica a viabilidade do problema

### Fase 2
- Remove variáveis artificiais do problema
- Otimiza a função objetivo original
- Retorna a solução ótima (se existir)

## Formato do Arquivo de Entrada

O arquivo de entrada deve seguir o formato:

```
MAX/MIN c1*x1 + c2*x2 + ... + cn*xn
a11*x1 + a12*x2 + ... + a1n*xn <= b1
a21*x1 + a22*x2 + ... + a2n*xn >= b2
a31*x1 + a32*x2 + ... + a3n*xn = b3
...
x1 >= 0
x2 <= 0
x3 livre
```

### Exemplo

```
MAX 3*x1 + 2*x2
x1 + x2 <= 4
2*x1 + x2 >= 3
x1 >= 0
x2 >= 0
```

## Como Usar

### Instalação

Nenhuma dependência externa é necessária. Apenas Python 3.x.

### Execução

1. Na função `main()`, altere a linha caso necessário passando o caminho do arquivo de entrada:
   ```python
   filename = "instancia5_lp.txt"
   ```
   Para o caminho do seu arquivo de entrada.

2. Na chamada de `generate_results_file()`, altere a linha caso necessário passando o caminho do arquivo de saída:
   ```python
   lp.generate_results_file("instancia5_lp_resultado.txt")
   ```
   Para o caminho onde deseja salvar o arquivo de saída.

Depois de configurar os caminhos, execute:

```bash
python simplex.py
```

Ou, se preferir passar o arquivo de entrada como argumento:

```bash
python simplex.py caminho/do/arquivo_entrada.txt
```

(Neste caso, o arquivo de saída ainda usará o caminho definido no código)

## Saída

O programa gera um arquivo de resultados contendo:

1. **Status da solução**: optimal, infeasible, unbounded, etc.
2. **Valor da função objetivo**
3. **Valores das variáveis** na solução ótima
4. **Verificação das restrições** (satisfeitas ou não)
5. **Histórico completo dos tableaus**:
   - Tableaus da Fase 1
   - Tableaus da Fase 2
   - Estado inicial e após cada iteração

## Estrutura do Código

### Classe Principal: `LinearProgram`

#### Métodos de Parsing
- `parse_file(filename)`: Lê e processa o arquivo de entrada
- `_parse_objective(line)`: Processa a função objetivo
- `_parse_constraint(line)`: Processa cada restrição
- `_parse_variable_domain(line)`: Processa domínios das variáveis

#### Transformação do Problema
- `transform_variables()`: Converte variáveis para forma padrão
- `build_standard_form()`: Constrói a forma padrão do PL

#### Resolução
- `phase_one()`: Executa a Fase 1 do Simplex
- `phase_two()`: Executa a Fase 2 do Simplex
- `run_simplex()`: Implementa o algoritmo Simplex
- `_pivot()`: Realiza operação de pivoteamento

#### Utilidades
- `calculate_objective_value()`: Calcula o valor da FO
- `verify_constraints()`: Verifica satisfação das restrições
- `generate_results_file()`: Gera arquivo com resultados completos

## Tratamento de Variáveis

### Variáveis Não-Negativas (`x >= 0`)
Mantidas como estão.

### Variáveis Não-Positivas (`x <= 0`)
Substituídas por `x_neg`, onde `x = -x_neg` e `x_neg >= 0`.

### Variáveis Livres (`x livre`)
Substituídas por `x = x_pos - x_neg`, onde `x_pos >= 0` e `x_neg >= 0`.

## Tratamento de Restrições

### Restrição `<=`
Adiciona variável de folga: `ax <= b` → `ax + s = b`, `s >= 0`

### Restrição `>=`
Adiciona variável de excesso e artificial: `ax >= b` → `ax - e + a = b`, `e, a >= 0`

### Restrição `=`
Adiciona variável artificial: `ax = b` → `ax + a = b`, `a >= 0`

## Possíveis Status de Saída

- `optimal`: Solução ótima encontrada
- `infeasible`: Problema inviável (sem solução)
- `unbounded`: Problema ilimitado (solução tende ao infinito)
- `pivot_failed`: Falha numérica durante pivoteamento
- `max_iterations`: Limite de iterações atingido

## Limitações e Considerações

- Precisão numérica: `EPS = 1e-9`
- Máximo de iterações: 5000 por fase
- Arquivo de entrada deve estar codificado em UTF-8
- Nomes de variáveis devem começar com letra e conter apenas letras, números e underscore

## Exemplo de Arquivo de Entrada

```
MAX 5*x1 + 4*x2
2*x1 + 3*x2 <= 12
3*x1 + 2*x2 <= 12
x1 >= 0
x2 >= 0
```

**Saída esperada para este exemplo:**
```
Status: optimal
Valor ótimo: 20.000000

Variáveis:
  x1 = 2.400000
  x2 = 2.400000

Verificação das Restrições:
  R1: 12.000000 <= 12.000000 ✓
  R2: 12.000000 <= 12.000000 ✓
```

## Licença

Copyright © 2025 Beatriz Nahas, João Plinta, Rafael Alves  
Todos os direitos reservados.

Este código é propriedade privada. Nenhuma parte deste software pode ser reproduzida, distribuída ou utilizada sem permissão prévia por escrito dos autores.

Para mais detalhes, consulte o arquivo [LICENSE](LICENSE).

## Contato

Para dúvidas ou sugestões, entre em contato com os autores.