#include <stdio.h>
#include <stdlib.h>
#define MAXWORDS 5000000

char words[MAXWORDS * 10]; //word length capped to 10

void readInput(int M, char *filename)
{
    FILE *ipf = fopen(filename, "r");
    int totalWordCount = 0;
    while( fscanf(ipf, "%s", &words[totalWordCount*10]) != EOF && totalWordCount < M && totalWordCount < MAXWORDS)
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

    readInput(M, filename);

    for(int i=0; i<M; i++)
        printf("%s\n", &words[i*10]);
    return 0;
}