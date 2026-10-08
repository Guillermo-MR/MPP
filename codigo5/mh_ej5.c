#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>

#include "../include/mh.h"

#define MUTATION_RATE 0.15
#define PRINT 0

//Variable global para la semilla, privada para cada hilo
unsigned int seed;
#pragma omp threadprivate(seed)

int aleatorio(int n) {
	return (rand_r(&seed) % n);  // genera un numero aleatorio entre 0 y n-1
}

int find_element(int *array, int end, int element)
{
	int i=0;
	int found=0;
	
	// comprueba que un elemento no está incluido en el individuo (el cual no admite enteros repetidos)
	while((i < end) && ! found) {
		if(array[i] == element) {
			found = 1;
		}
		i++;
	}
	return found;
}

int *crear_individuo(int n, int m)
{
	int i=0, value;
	int *individuo = (int *) malloc(m * sizeof(int));
	
	// inicializa array de elementos
	memset(individuo, -1, m * sizeof(int));
	
	while(i < m) {
		value = aleatorio(n);
		// si el nuevo elemento no está en el array...
		if(!find_element(individuo, i, value)) {
			individuo[i] = value;  // lo incluimos
			i++;
		}
	}
	return individuo;
}

int comp_array_int(const void *a, const void *b) {
	return (*(int *)a - *(int *)b);
}

int comp_fitness(const void *a, const void *b) {
	/* qsort pasa un puntero al elemento que está ordenando */
	return (*(Individuo **)b)->fitness - (*(Individuo **)a)->fitness;
}

double aplicar_mh(const double *d, int n, int m, int n_gen, int tam_pob, int *sol)
{
        // para reiniciar la secuencia pseudoaleatoria en cada ejecución
        //srand(time(NULL) + getpid());
	int i, g, mutation_start;
	
	// crea poblacion inicial (array de individuos)
	Individuo **poblacion = (Individuo **) malloc(tam_pob * sizeof(Individuo *));
	assert(poblacion);
	
	//Descativamos el ajuste dinamico de hilos
	omp_set_dynamic(0);
	
	//Región paralela, abrimos el bloque antes de FOR_INI
        #pragma omp parallel
        {
          /*Cada hilo inicializa su propia semilla basándose en el tiempo, el PID del proceso y su identificador de hilo único */
          seed = time(NULL) + getpid() + omp_get_thread_num();
          
          //Repartimos el bucle FOR_INI entre los hilos generados
          // static: todas las iteraciones cuestan practicamente lo mismo
          #pragma omp for schedule(static)
          for(i = 0; i < tam_pob; i++) {
              poblacion[i] = (Individuo *) malloc(sizeof(Individuo));
              poblacion[i]->array_int = crear_individuo(n, m);
              
              // calcula el fitness del individuo
	      fitness(d, poblacion[i], n, m);
          }
        }
	
	// ordena individuos segun la funcion de bondad (mayor "fitness" --> mas aptos)
	qsort(poblacion, tam_pob, sizeof(Individuo *), comp_fitness);
	
	// evoluciona la poblacion durante un numero de generaciones
	for(g = 0; g < n_gen; g++)
	{
		// los hijos de los ascendientes mas aptos sustituyen a la ultima mitad de los individuos menos aptos
		// Cada iteracion lee dos padres de la primera mitad y escribe dos hijos distintos de la segunda,
		// asi que las iteraciones son independientes
		// dynamic,1: el coste de cada cruce depende del punto de corte aleatorio
		#pragma omp parallel for schedule(dynamic,1)
		for(i = 0; i < (tam_pob/2) - 1; i += 2) {
			cruzar(poblacion[i], poblacion[i+1], poblacion[tam_pob/2 + i], poblacion[tam_pob/2 + i + 1], n, m);
		}
		
		// inicia la mutacion a partir de 1/4 de la poblacion
		mutation_start = tam_pob/4;
		
		// muta 3/4 partes de la poblacion
		for(i = mutation_start; i < tam_pob; i++) {
			mutar(poblacion[i], n, m);
		}
		
		// recalcula el fitness del individuo
		for(i = 0; i < tam_pob; i++) {
			fitness(d, poblacion[i], n, m);
		}
		
		// ordena individuos segun la funcion de bondad (mayor "fitness" --> mas aptos)
		qsort(poblacion, tam_pob, sizeof(Individuo *), comp_fitness);
		
		if (PRINT) {
			printf("Generacion %d - ", g);
			printf("Fitness = %.0lf\n", poblacion[0]->fitness);
		}
	}
	
	// ordena el array solucion
	qsort(poblacion[0]->array_int, m, sizeof(int), comp_array_int);

        // y lo mueve a sol para escribirlo
	memmove(sol, poblacion[0]->array_int, m*sizeof(int));
	
	// almacena el mejor valor obtenido para el fitness
	double value = poblacion[0]->fitness;
		
	// se libera la memoria reservada
	
	for (i = 0; i < tam_pob; i++) {
    	    free(poblacion[i]->array_int);
    	    free(poblacion[i]);
	}
	
	free(poblacion);
	
	// devuelve el valor obtenido para el fitness
	return value;
}

void cruzar(Individuo *padre1, Individuo *padre2, Individuo *hijo1, Individuo *hijo2, int n, int m)
{
	// Elegir un "punto" de corte aleatorio a partir del que se realiza el intercambio de los genes
	
	// Los primeros genes del padre1 van al hijo1. Idem para el padre2 e hijo2.
	
	// Y los restantes son del otro padre, respectivamente.
	
	// Factibilizar: eliminar posibles repetidos de ambos hijos
	// Si encuentro alguno repetido en el hijo1, lo cambio por otro que no este en el conjunto
	int corte = 1 + aleatorio(m-1);
	
	for(int i = 0; i < m; i++){
	
	    if(i < corte){
	        
	        hijo1->array_int[i] = padre1->array_int[i];
	        hijo2->array_int[i] = padre2->array_int[i];
	        
	    }else{
	        int v1 = padre2->array_int[i];
	        int v2 = padre1->array_int[i];
	        
	        while(find_element(hijo1->array_int, i, v1))
	            v1 = aleatorio(n);
	        
	        while(find_element(hijo2->array_int, i, v2))
	            v2 = aleatorio(n);
	            
	        hijo1->array_int[i] = v1;
	        hijo2->array_int[i] = v2;
	    }
	}
	
	
	
}

void mutar(Individuo *actual, int n, int m)
{
	// Decidir cuantos elementos mutar:
	// Si el valor es demasiado pequeño la convergencia es muy pequeña y si es demasiado alto diverge
	
	// Cambia el valor de algunos elementos de array_int de forma aleatoria
	// teniendo en cuenta que no puede haber elementos repetidos:
        // una posibilidad podría ser usar una variable, m_rate, para establecer la intensidad de la mutación 
        // (un bucle for con un número de iteraciones que dependa, por ejemplo, de m_rate*m)
        
        
        
        for(int i = 0; i < m; i++){
            // ocurre con probabilidad 0.15 hasta que pongamos la variable en el proximo ejercicio
            if (rand_r(&seed) / (RAND_MAX + 1.0) < MUTATION_RATE) {
                int v = aleatorio(n);
                while(find_element(actual->array_int, m, v))
                    v = aleatorio(n);
                actual->array_int[i] = v;
            }
        }
        
}

double distancia_ij(const double *d, int i, int j, int n)
{
        // Devuelve la distancia entre dos elementos i, j de la matriz 'd'
        // 'd' es la forma compacta D' (solo el triangulo superior, i < j), asi que si i > j
        // se intercambian, ya que d_ij = d_ji
        if (i > j) {
                int aux = i;
                i = j;
                j = aux;
        }

        // indice k = f(i, j, n) del enunciado:
        // k = (n^2 - n)/2 - ((n-i)^2 - (n-i))/2 + j - i - 1
        int k = (n*n - n)/2 - ((n-i)*(n-i) - (n-i))/2 + j - i - 1;

        return d[k];
}

void fitness(const double *d, Individuo *individuo, int n, int m)
{
	// Determina la calidad del individuo calculando la suma de la distancia entre cada par de enteros
	double suma = 0.0;
	int a, b;

	// Compartidas: d, individuo, n, m (solo lectura)
	// Privadas: a (indice del bucle paralelizado, privada automaticamente) y b
	// Reduccion: cada hilo acumula en una copia privada de suma inicializada a 0
	// y al terminar el bucle las copias se suman sobre la variable original
	#pragma omp parallel for default(none) shared(d, individuo, n, m) private(b) reduction(+:suma)
	// cada par (a, b) con a < b se suma una sola vez
	for (a = 0; a < m - 1; a++) {
		for (b = a + 1; b < m; b++) {
			suma += distancia_ij(d, individuo->array_int[a], individuo->array_int[b], n);
		}
	}

	individuo->fitness = suma;
}
