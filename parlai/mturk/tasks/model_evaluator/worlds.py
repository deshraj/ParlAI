# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.worlds import validate, create_task
from parlai.mturk.core.worlds import MTurkTaskWorld, MTurkOnboardWorld

NUM_DIALOG_ROUNDS = 10

class ModelEvaluatorOnboardWorld(MTurkOnboardWorld):
    def parley(self):
        ad = {}
        ad['id'] = 'System'
        ad['text'] = 'Welcome onboard! Enter anything to confirm you\'re here.'
        self.mturk_agent.observe(ad)
        self.mturk_agent.act()
        self.episodeDone = True


class ModelEvaluatorWorld(MTurkTaskWorld):
    """World for letting Turkers evaluate a dialog model's performance given a
    context. Assumes the context is a context from a given task, e.g.
    from SQuAD, CBT, etc.
    """

    evaluator_agent_id = 'Model Evaluator'

    def __init__(self, opt, model_agent, task_opt, mturk_agent):
        self.task_world = create_task(task_opt, model_agent)
        self.mturk_agent = mturk_agent
        self.episodeDone = False
        self.round_id = 0

    def parley(self):
        self.task_world.parley()

        ad = {}
        # Show the dialog model's response to the context, and ask the turker
        # to rate the response
        ad['id'] = self.__class__.evaluator_agent_id

        # Show a random answer after each round of dialog
        ad['text'] = ("Answer {0}".format(self.round_id))

        global NUM_DIALOG_ROUNDS

        if (self.round_id == NUM_DIALOG_ROUNDS):
            ad['episode_done'] = True  # self.world.episode_done()
            self.episodeDone = True
        else:
            ad['episode_done'] = False  # self.world.episode_done()
            self.round_id += 1

        self.mturk_agent.observe(validate(ad))

        ad['text'] = ("Please ask the next question based on previous context.")
        self.mturk_agent.observe(validate(ad))
        self.rating = self.mturk_agent.act()
        # Can log the rating here

    def episode_done(self):
        return self.episodeDone

    def report(self):
        pass

    def shutdown(self):
        self.task_world.shutdown()
        self.mturk_agent.shutdown()

    def review_work(self):
        pass
