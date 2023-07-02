import pygame
from pong import Game

import os
import neat

class Pong:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)

        self.leftPaddle = self.game.left_paddle
        self.rightPaddle = self.game.right_paddle
        self.ball = self.game.ball

    def testAi(self):

        run = True

        clock = pygame.time.Clock()

        while run:
            
            clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
            
            keys = pygame.key.get_pressed()

            if keys[pygame.K_w]:
                game.move_paddle(left=True, up=True)

            if keys[pygame.K_s]:
                game.move_paddle(left=True, up=False)

            gameInfo = game.loop()
            game.draw()

            pygame.display.update()

        pygame.quit()
    
    def trainAi(self, genome1, genome2, config):

        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
            
            net1Output = net1.activate((self.leftPaddle.y, self.ball.y, abs(self.leftPaddle.x - self.ball.x)))
            move1 = net1Output.index(max(net1Output))

            net2Output = net2.activate((self.rightPaddle.y, self.ball.y, abs(self.rightPaddle.x - self.ball.x)))
            move2 = net2Output.index(max(net2Output))

            if move1 == 0:
                pass
            elif move1 == 1:
                self.game.move_paddle(left=True, up=True)
            else:
                self.game.move_paddle(left=True, up=False)

            if move2 == 0:
                pass
            elif move2 == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)




            gameInfo = self.game.loop()
            self.game.draw(draw_score=False, draw_hits=True)
            pygame.display.update()

            if gameInfo.left_score >= 1 or gameInfo.right_score >= 1 or gameInfo.left_hits > 50:
                self.calculateFitness(genome1, genome2, gameInfo)
                break
    
    def calculateFitness(self, genome1, genome2, gameInfo):
        genome1.fitness += gameInfo.left_hits
        genome2.fitness += gameInfo.right_hits


def evalGenomes(genomes, config):

    width , height = 800, 600
    window = pygame.display.set_mode((width, height))

    for i , (genomeId1 , genome1) in enumerate(genomes):
        
        if i == len(genomes) - 1:
            break
        
        genome1.fitness = 0

        for genomeId2 , genome2 in genomes[i+1:]:

            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness

            game = Pong(window,width, height)

            game.trainAi(genome1, genome2, config)

            

def runNeat(config):
    #p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-0")
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(1))

    winner = population.run(evalGenomes, 50)


if __name__ == "__main__":
    localDir = os.path.dirname(__file__)
    configPath = os.path.join(localDir, "config.txt")

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, configPath)

    runNeat(config)

