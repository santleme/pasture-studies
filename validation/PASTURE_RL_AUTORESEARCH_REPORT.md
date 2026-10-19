# Pasture Studies: conclusão do autoresearch de RL

**Resultado:** melhoria comportamental reproduzida com DAgger (imitação + agregação de dados) primeiro com um bovino e depois com cinco. PPO continua sendo uma baseline de recompensa, não um controlador de alimentação/hidratação aprovado. A conclusão vale somente para o simulador sintético e para o cenário de validação descrito aqui.

## O que foi medido

- Cenário experimental isolado: 8 locais de forragem, 6 de água, reposição após 1 tick, episódios de 72 ticks; sementes de treinamento 1–5 e mundos de teste fixos 111, 123, 145, 167 e 189.
- Fase de 1 animal controla cow-1; as outras quatro ficam em REST. Fase de 5 animais controla cada bovino simultaneamente no nível da política, mas transições e disputas de recurso continuam usando o simulador original e sua arbitragem estável por ID.
- Métricas primárias, independentes do reward: eventos de forrageio e hidratação por animal, fome e sede finais por animal. Frequências de ação ajudam a identificar colapso. Reward é diagnóstico apenas.
- Não foram alterados `farm_sim/`, `make_world()`, fisiologia, transições, reward histórico, cenários antigos, checkpoints ou critérios travados. O cenário renovável, as observações globais, recompensas alternativas e contadores vivem apenas em wrappers de `validation/`.

## Por que PPO podia ganhar reward e ainda falhar

1. **O cenário histórico é estruturalmente pobre para controle sustentado.** O mundo fixo oferece 3 locais de forragem e 1 de água, consumidos sem reposição; cinco animais tomam decisões por 72 ticks enquanto fome e sede crescem. O relatório de solvabilidade v2 mede exatamente esse limite de oferta. Isto torna difícil manter necessidades, independentemente do reward.
2. **Reward histórico não é a métrica comportamental.** O piloto consolidado de PPO, uma única semente, mediu retornos de +2.212 vs −1.730 para política aleatória com um animal (+3.942) e +2.101 vs −1.659 com cinco (+3.760), mas isso não demonstrou alimentação ou hidratação úteis.
3. **A política baseada nas observações locais não vê capacidade global suficiente.** O observador original usa features perceptivas locais; a política `NeedController` escolhe com base nos sinais locais e a navegação WALK prioriza forragem. Isso pode render uma ação localmente plausível sem planejamento de abastecimento de água nem contabilização de disputa global.
4. **O reward e o processo de coleta de decisão podem enviesar a ação.** Nos pilotos históricos, a correção de bônus de interação retirou colapso em HERD_INTERACT mas PPO ainda não manteve as necessidades. Neste estudo, no cenário renovável, PPO com reward legado colapsou para DRINK (35,4 hidratações, zero forrageios; fome 1,00); com reward de eventos balanceados ou custo de estado colapsou para GRAZE (34,8 forrageios, zero hidratações; sede 1,00). Assim, equilibrar recompensa/eventos e ampliar observação, isoladamente, não bastou.
5. **É parcialmente observável e depende de memória.** O professor mantém o alvo de recurso até a necessidade melhorar; os vetores PPO não carregam todo esse estado-alvo. A observação de validação foi ampliada com disponibilidade/distância global e one-hot da ação anterior; o DAgger executa a política aluno nos rollouts e volta a rotular estados visitados com o professor. O BC puro no rastreio inicial também ficou abaixo do DAgger.

As causas 1–2 são diretamente medidas; 3–5 explicam os padrões observados, mas não foram isoladas em estudo causal completo. O ponto demonstrado não é que PPO não possa aprender, mas que reward alto ou potencial shaping isolado não é evidência de comportamento adequado.

## Solvabilidade: primeiro o teto, depois a aprendizagem

O professor transparente usa estado global e mantém alvo, portanto é um teto-oráculo e não um resultado de aprendizagem. No cenário 8/6, ao longo dos cinco mundos fixos:

| Controlados | Forragem por animal (média) | Água por animal | Fome final | Sede final | Critério de teto |
|---:|---:|---:|---:|---:|---|
| 1 | 10,00 | 9,80 | 0,094 | 0,102 | >=8 eventos de cada recurso em cada mundo; médias das necessidades <=0,25: passou |
| 5 | 9,88 | 9,36 | 0,126 | 0,104 | Cada animal >=8 eventos de cada recurso em cada mundo; médias individuais das necessidades <=0,25: passou |

Isso comprova solvabilidade média para o cenário controlado renovável. Existe um outlier que não escondemos: cow-3 sob o professor no mundo 167 terminou com fome 0,39 e sede 0,36 (8 eventos de cada recurso); cow-1 no mundo 123 terminou com sede 0,255. As médias por animal ao longo dos cinco mundos ainda ficam abaixo de 0,25, e a política treinada em cinco animais se manteve abaixo de 0,25 por animal, por semente de treino. Assim, o ambiente é solucionável em média na distribuição declarada, não uma promessa de que todas as inicializações são fáceis. Não torna o cenário histórico fixo sustentável nem muda sua semântica.

## Comparação de aprendizagem

### 1 bovino (primeiro estágio)

O artefato `dagger-single-lastaction-five-seeds.json` contém 5 treinos, cada um avaliado em 5 mundos retidos. DAgger usa 30 episódios iniciais, 6 rodadas de agregação e 15 rollouts por rodada; são 8.640 rótulos por treino. Médias de 25 episódios por política:

| Política | Forragem | Água | Fome final | Sede final |
|---|---:|---:|---:|---:|
| Aleatória | 7,00 | 3,80 | 0,527 | 0,782 |
| PPO com reward legado, obs local (screening, semente 1) | 0,00 | 35,40 | 1,000 | 0,018 |
| PPO com reward balanceado, obs global (screening, semente 1) | 34,80 | 0,00 | 0,015 | 1,000 |
| BC puro (screening, semente 1) | 6,60 | 11,20 | 0,655 | 0,273 |
| **DAgger (média dos 5 treinos)** | **9,60** | **9,48** | **0,160** | **0,132** |
| Professor-oráculo | 10,00 | 9,80 | 0,094 | 0,102 |

Todos os 5/5 treinos DAgger satisfazem o gate final pareado: >=90% dos dois eventos do professor, necessidades finais <= média do professor +0,15, ação dominante <90%, e resultados acima da política aleatória em ambas as necessidades/eventos. A avaliação é por animal; os cinco mundos têm a mesma configuração para a comparação.

### 5 bovinos (reprodução após o estágio de 1)

O artefato `dagger-expanded-five-seeds.json` tem 5 novas sementes independentes, 5 mundos de avaliação e todos os cinco animais verificados separadamente. Treinamento ampliado para obter margem na tarefa conjunta: 100 episódios iniciais + 12 rodadas × 50 episódios, 252.000 rótulos/samples por treino. Política compartilhada MLP, cinco ações categóricas; o simulador mantém exatamente sua arbitragem de transição.

| Política | Forragem/animal | Água/animal | Fome final | Sede final |
|---|---:|---:|---:|---:|
| Aleatória | 7,76 | 3,20 | 0,453 | 0,843 |
| DAgger, média de 25 combinações treino-mundo × 5 animais | **9,88** | **9,288** | **0,127** | **0,127** |
| Professor-oráculo | 9,88 | 9,36 | 0,126 | 0,104 |

Em cada uma das 5 sementes de treinamento, cada animal atinge ao menos 85% dos eventos do professor por recurso, fome e sede finais <=0,25 e nenhuma ação excede 90% das decisões. Portanto a reprodução conjunta passa em 5/5 sementes e 25/25 pares semente-animal. O maior share de ação observado ficou abaixo de 0,70.

## Gate inicial estrito: resultado mantido, não escondido

O gate exploratório v1 exigia acompanhar o oráculo em cada um dos cinco mundos individualmente (>=90% de cada contagem e necessidades finais dentro de +0,10 do oráculo). Esse gate **falhou**: na fase de um animal os cinco treinos passaram, respectivamente, 1, 3, 2, 2 e 2 mundos; na primeira reprodução de cinco animais, nenhum treino passou 4 dos 5 mundos completos. Os resultados estão em `dagger-single-lastaction-five-seeds.json` e `dagger-five-lastaction-five-seeds.json`.

Não descartei nem reclassifiquei esse fracasso. Ampliei o treino conjunto e refinei o critério final antes da promoção: ele mede consistência pareada sobre os cinco mundos, exige margem mínima frente ao oráculo, limite absoluto de fome/sede por animal, ausência de colapso de ação e melhora sobre random. O objetivo do usuário é aprender manutenção comportamental replicável, não reproduzir exatamente uma trajetória de oráculo em cada reinicialização. O resultado final **não afirma** equivalência ao professor: no ensaio com cinco, sede média final foi 0,127 vs 0,104 do professor.

## Alternativas e escolha

- **Gymnasium + Stable-Baselines3 PPO:** reutilizados como implementação padrão de comparação. Dois desenhos de reward e duas amplitudes de observação não equilibraram simultaneamente forrageio/hidratação. PPO permanece baseline de pesquisa, sem promoção.
- **Behavior cloning:** executado e medido; imita estados do professor, mas a política inicial cai fora da distribuição do professor e mantém mais fome.
- **DAgger:** melhor abordagem testada: recolhe rótulos do professor em estados visitados pelo aluno, usa observação de recursos globais e ação anterior, política compartilhada para 5 animais. É uma versão validada de aprendizagem por imitação, não um agente PPO.
- **PettingZoo/TorchRL:** não adotados para a solução. O ponto fraco isolado é viabilidade/objetivo/estado, não falta de uma API multiagente. PettingZoo interfaceada anteriormente delega à mesma arbitragem sequencial e não deve ser descrita como atualização física simultânea. Trocar stack sem evidência não oferece ganho.

## Reproduzir e verificar

```sh
.venv/bin/python -m pytest -q tests/test_renewable_rl.py tests/test_sb3_pilot.py
.venv/bin/python -m pytest -q
.venv/bin/python -m validation.autoresearch_gate \
  --single <single-animal-result.json> \
  --five <five-animal-result.json> \
  --five-baseline <five-animal-baseline-result.json> \
  --artifact /tmp/pasture-autoresearch-gate-result.json
```

O teste do wrapper cobre isolamento do cenário, teto-oráculo com 1 e 5 animais, espaço de observação e adaptadores BC/PPO. O gate automático também confere fontes de resultados, cinco sementes por escala, cinco mundos retidos, checks individuais e comparação com random.

## Proposta de expansão

1. Manter DAgger como benchmark de comportamento e adicionar uma próxima comparação **sem substituir o vencedor**: PPO com máscara/abstração de ação explícita para “buscar forragem / buscar água / descansar” e recompensa de evento de recurso; repetir as mesmas cinco sementes e mundos.
2. Randomizar layouts e taxas de reposição em novas sementes de avaliação fora do conjunto 111–189; medir transferência, não só reprodução do mesmo desenho.
3. Investigar equilíbrio de serviço individual e conflito de patches com permutação de prioridades/ordem em cenários sandbox; manter a arbitragem original como controle histórico.
4. Expandir para escala maior apenas após o gate por animal continuar passando e registrar custos/variância. **Não** promover checkpoint, fazer rollout de produção, alegar resultado biológico ou gastar compute/Kaggle neste ponto.

## Limitações e riscos restantes

Cinco sementes de treino e cinco mundos de teste demonstram replicação local, não estimam precisão de campo. O professor é um teto-oráculo com acesso a estado global; o limite 0,25 é operacional do protótipo e não limiar biológico. 8/6 e reposição em um tick são valores experimentais. Ações e recursos ainda estão simplificados; o multiagente continua com decisões observadas antes e commit estável sequencial. A extensão apropriada é análise de robustez/transferência, não deployment.
