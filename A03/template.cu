#include <stdio.h>
#include <stdlib.h>
#define MAXWORDS 5000000

char *words; //word length capped to 10

void readInput(int M, char *filename)
{
    FILE *ipf = fopen(filename, "r");
    int totalWordCount = 0;
    while(totalWordCount < M && totalWordCount < MAXWORDS && fscanf(ipf, "%s", &h_words[totalWordCount*10]) != EOF)
    {
        totalWordCount++;
    }
    fclose(ipf);
}

int main(int argc, char *argv[])
{
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    char *filename = argv[3];

    words = (char*)malloc(sizeof(char)*10*M);
    readInput(M, filename);

    for(int i=0; i<M; i++)
        printf("%s -- %d\n", &words[i*10], (int)strlen(&words[i*10]));
    return 0;
}