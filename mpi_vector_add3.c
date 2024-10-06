/* File:     mpi_vector_add.c
 *
 * Purpose:  Implement parallel vector addition using a block
 *           distribution of the vectors.  This version also
 *           illustrates the use of MPI_Scatter and MPI_Gather.
 *
 * Compile:  mpicc -g -Wall -o mpi_vector_add mpi_vector_add.c
 * Run:      mpiexec -n <comm_sz> ./vector_add
 *
 * Input:    The order of the vectors, n, and the vectors x and y
 * Output:   The sum vector z = x+y
 *
 * Notes:
 * 1.  The order of the vectors, n, should be evenly divisible
 *     by comm_sz
 * 2.  DEBUG compile flag.
 * 3.  This program does fairly extensive error checking.  When
 *     an error is detected, a message is printed and the processes
 *     quit.  Errors detected are incorrect values of the vector
 *     order (negative or not evenly divisible by comm_sz), and
 *     malloc failures.
 *
 * IPP:  Section 3.4.6 (pp. 109 and ff.)
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

void Check_for_error(int local_ok, char fname[], char message[], MPI_Comm comm);
void Allocate_vectors(double** local_x_pp, double** local_y_pp, double** local_z_pp, int local_n, MPI_Comm comm);
double Parallel_dot_product(double local_x[], double local_y[], int local_n);
void Parallel_scalar_multiplication(double local_x[], double local_y[], double scalar, int local_n);
void Print_vector(double local_b[], int local_n, int n, char title[], int my_rank, MPI_Comm comm);

int main(void) {
   int n = 20;
   int local_n;
   int comm_sz, my_rank;
   double *local_x, *local_y, *local_z;
   double dot_product;
   double scalar = 2.5;
   MPI_Comm comm;
   double tstart, tend;

   MPI_Init(NULL, NULL);
   comm = MPI_COMM_WORLD;
   MPI_Comm_size(comm, &comm_sz);
   MPI_Comm_rank(comm, &my_rank);

   local_n = n / comm_sz;
   
   tstart = MPI_Wtime();
   Allocate_vectors(&local_x, &local_y, &local_z, local_n, comm);

   srand(time(NULL) + my_rank); // Para generar números aleatorios diferentes en cada proceso
   for (int i = 0; i < local_n; i++) {
      local_x[i] = (double)(rand() % 100);
      local_y[i] = (double)(rand() % 100);
   }

   MPI_Barrier(comm); // Sincronización

   // Imprimir los vectores antes de la multiplicación
   Print_vector(local_x, local_n, n, "Vector x (antes de multiplicar por escalar)", my_rank, comm);
   Print_vector(local_y, local_n, n, "Vector y (antes de multiplicar por escalar)", my_rank, comm);

   // Calcular producto punto
   dot_product = Parallel_dot_product(local_x, local_y, local_n);

   // Multiplicación por un escalar
   Parallel_scalar_multiplication(local_x, local_y, scalar, local_n);
   Print_vector(local_x, local_n, n, "Vector x (después de multiplicar por escalar)", my_rank, comm);
   Print_vector(local_y, local_n, n, "Vector y (después de multiplicar por escalar)", my_rank, comm);

   if (my_rank == 0) {
      printf("Producto punto: %f\n\n", dot_product);
   }

   tend = MPI_Wtime();

   if (my_rank == 0) {
      printf("Tiempo total: %f milisegundos\n", (tend - tstart) * 1000);
   }

   free(local_x);
   free(local_y);
   free(local_z);

   MPI_Finalize();
   return 0;
}

void Allocate_vectors(double** local_x_pp, double** local_y_pp, double** local_z_pp, int local_n, MPI_Comm comm) {
   *local_x_pp = (double*) malloc(local_n * sizeof(double));
   *local_y_pp = (double*) malloc(local_n * sizeof(double));
   *local_z_pp = (double*) malloc(local_n * sizeof(double));
   if (*local_x_pp == NULL || *local_y_pp == NULL || *local_z_pp == NULL) {
      fprintf(stderr, "Error en la asignación de memoria\n");
      MPI_Abort(comm, -1);
   }
}

double Parallel_dot_product(double local_x[], double local_y[], int local_n) {
   double local_dot_product = 0.0;
   double global_dot_product = 0.0;
   for (int i = 0; i < local_n; i++) {
      local_dot_product += local_x[i] * local_y[i];
   }
   MPI_Reduce(&local_dot_product, &global_dot_product, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   return global_dot_product;
}

void Parallel_scalar_multiplication(double local_x[], double local_y[], double scalar, int local_n) {
   for (int i = 0; i < local_n; i++) {
      local_x[i] *= scalar;
      local_y[i] *= scalar;
   }
}

void Print_vector(double local_b[], int local_n, int n, char title[], int my_rank, MPI_Comm comm) {
   double* b = NULL;
   if (my_rank == 0) {
      b = malloc(n * sizeof(double));
   }
   MPI_Gather(local_b, local_n, MPI_DOUBLE, b, local_n, MPI_DOUBLE, 0, comm);
   if (my_rank == 0) {
      printf("%s:\n", title);
      for (int i = 0; i < n; i++) {
         printf("%f ", b[i]);
      }
      printf("\n");
      free(b);
   }
}