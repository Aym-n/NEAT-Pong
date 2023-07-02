import pygame
from pong import Game

import os
import neat

import pickle

class Pong:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        pygame.display.set_caption('AI Pong')

        self.leftPaddle = self.game.left_paddle
        self.rightPaddle = self.game.right_paddle
        self.ball = self.game.ball

    def testAi(self , genome , config):

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        run = True

        clock = pygame.time.Clock()

        while run:
            
            clock.tick(120)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
            
            keys = pygame.key.get_pressed()

            if keys[pygame.K_UP]:
                self.game.move_paddle(left=False, up=True)

            if keys[pygame.K_DOWN]:
                self.game.move_paddle(left=False, up=False)
            
            if keys[pygame.K_c]:
                self.game.reset()


            Output = net.activate((self.leftPaddle.y, self.ball.y, abs(self.leftPaddle.x - self.ball.x)))
            move = Output.index(max(Output))

            if move == 0:
                pass
            elif move == 1:
                self.game.move_paddle(left=True, up=True)
            else:
                self.game.move_paddle(left=True, up=False)

            gameInfo = self.game.loop()
            self.game.draw(True, False)

            pygame.display.update()

        pygame.quit()

    def testAiVsAi(self , genome , config):

        net1 = neat.nn.FeedForwardNetwork.create(genome, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome, config)

        run = True

        clock = pygame.time.Clock()

        while run:

            clock.tick(300)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            Output1 = net1.activate((self.leftPaddle.y, self.ball.y, abs(self.leftPaddle.x - self.ball.x)))
            move1 = Output1.index(max(Output1))

            Output2 = net2.activate((self.rightPaddle.y, self.ball.y, abs(self.rightPaddle.x - self.ball.x)))
            move2 = Output2.index(max(Output2))

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
            self.game.draw(False, True)

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
    population = neat.Checkpointer.restore_checkpoint("neat-checkpoint-49")
    #population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(1))

    winner = population.run(evalGenomes, 50)

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)

def testAi(config , mode):
    with open("winner.pkl", "rb") as f:
        genome = pickle.load(f)
    
    width , height = 800, 600
    window = pygame.display.set_mode((width, height))

    game = Pong(window, width, height)
    if(mode):
        game.testAiVsAi(genome, config)

    else:
        game.testAi(genome, config)

if __name__ == "__main__":
    localDir = os.path.dirname(__file__)
    configPath = os.path.join(localDir, "config.txt")

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, configPath)

    #runNeat(config)
    testAi(config, False)

