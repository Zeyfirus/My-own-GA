import numpy as np
data=open('input.txt','r')
result=open('output.txt','w')
task_count=int(data.readline())
task_coef=list(map(int,data.readline().split()))
time=list(map(float,data.readline().split()))
dev_count=int(data.readline())
dev_coef=np.array([])
for i in range(dev_count):
    dev_coef=np.append(dev_coef,list(map(float,data.readline().split())))
dev_coef=dev_coef.reshape(dev_count,4)
class GA:
    def __init__(self,task_count,task_coef,time,dev_count,dev_coef):
        self.task = task_count # Количество задач
        self.t_coef = task_coef # Уровень сложности каждой задачи
        self.time = time # Время для каждой задачи
        self.dev = dev_count # Количество разработчиков
        self.dev_coef = dev_coef # Коэффициенты разработчиков
        self.population_size=500 # Размер популяции
        self.selected_size=350
        self.fit=None
        self.population = self.create_population() # Начальная популяция
        self.n_steps=50 #Поколения
        self.p=0.15
    def create_population(self):
        '''Рондомна рапределяем разработчиков по задачам и
                формируем начальную популяцию (массив из номеров разработчиков)'''
        population = np.array([], dtype=int)
        count = 0
        while count < self.population_size:
            individual = np.random.randint(1, self.dev + 1, self.task)
            population = np.append(population, individual, axis=0)
            count += 1
        return population.reshape(self.population_size, self.task)

    def fitness(self, pop):
        if self.fit is not None:
            return self.fit
        def fitness_individ(individual):
            chrom = np.array([], dtype=float)
            count = 0
            for row, col in zip(individual, self.t_coef):
                chrom = np.append(chrom, self.dev_coef[row - 1, col - 1] * self.time[count])
                count += 1
            return np.max(chrom)
        fit = np.array([], dtype=float)
        for individual in pop:
            fit = np.append(fit, fitness_individ(individual))
        self.fit = fit
        return self.fit

    def selection(self):
        '''Турнирный отбор'''
        new_pop = np.array([], dtype=int)
        self.fitness(self.population)
        while len(new_pop) < self.selected_size*self.task:
            ind1,ind2,ind3,ind4 = np.random.choice(self.population_size,4,replace=False)
            best_ind = min(self.fit[int(ind1)],self.fit[int(ind2)],self.fit[int(ind3)],self.fit[int(ind4)])
            index = np.argwhere(self.fit == best_ind)
            new_pop = np.append(new_pop, self.population[index[0]])
        return new_pop.reshape(self.selected_size, self.task)

    def crossover(self):
        '''k-point скрещивание'''
        p = 0.7
        all = self.selection()
        new_gen = np.array([], dtype=int)
        for i in range(self.population_size // 2):
            points=np.random.choice(self.task,size=self.task)
            a_ind, b_ind = np.random.choice(self.selected_size, size=2, replace=False)
            prob = np.random.rand()
            a, b = all[a_ind], all[b_ind]
            if prob <= p:
                for ind in range(1, len(points) + 1):
                    if ind%2 == 1:
                        new_a = np.append(a[:points[ind - 1] + 1], b[points[ind - 1] + 1:])
                        new_b = np.append(b[:points[ind - 1] + 1], a[points[ind - 1] + 1:])
                    elif ind%2 == 0:
                        new_a = np.append(a[:points[ind - 1]], b[points[ind - 1]:])
                        new_b = np.append(b[:points[ind - 1]], a[points[ind - 1]:])
                new_pop = np.concatenate((new_a, new_b))
                new_gen = np.concatenate((new_gen, new_pop))
            else:
                new_pop = np.concatenate((a, b))
                new_gen = np.concatenate((new_gen, new_pop))
        return new_gen.reshape(self.population_size, self.task)


    def mutation(self):
        new = self.crossover()
        probs = np.random.rand(self.population_size) <= self.p
        for i, p in zip(np.arange(self.population_size), probs):
            if p:
                j = np.random.randint(self.task)
                new[i, j] = np.random.choice(self.dev)
        return new
    def generation(self):
        for i in range (self.n_steps):
            self.population = self.mutation()
            self.fit = None
        best_individ = self.population[np.argsort(self.fitness(self.population))][0]
        return best_individ

ga=GA(task_count,task_coef,time,dev_count,dev_coef)
result.write(str(ga.generation()))



