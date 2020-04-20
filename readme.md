O programa redhood.c, originalmente chamado bpsim.c foi baixado do seguinte link:
http://lcn.epfl.ch/tutorial/english/mlpc/html/index.html

A aplicação implementa uma rede neural perceptron de multi-camadas.
O usuário descreve um personagem do conto chapeuzinho vermelho e a aplicação tem como saída a atitude do personagem chapéuzinho vermelho diante do personagem. 
Ou seja, ele identifica o personagem descrito pelo usuário, as opções de personagem são: Vovózinha, Lobo e Caçador.

Para teste da paralelização deve se compilar o arquivo redhood.c com gcc:
gcc redhood.c -o redhood -lm -w -fopenmp

Em sequencia executá-lo utilizando o redirecionamento de entrada do arquivo "in", assim a aplicação irá executar a função de aprendizagem com 1000 iterações.
time ./redhood < in

O arquivo "in" encontra-se em anexo e contém o seguinte conteúdo:

Learn
1000
Quit




