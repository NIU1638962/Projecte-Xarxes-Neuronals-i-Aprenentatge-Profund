# -*- coding: utf-8 -*- noqa
"""
Created on Sun May  4 13:34:52 2025

@author: Alejandro
"""
from abc import ABC

import environment
import utils


def metric_method(func):
    """
    Decorator that tags a function as a metric method.
    Used to identify which methods should be registered in other classes
    containing the basic BinaryClassMetrics
    """
    func.is_metric = True
    return func


class MetricsClass(ABC):
    pass


class BinaryClassMetrics(MetricsClass):
    __slots__ = ("__tp", "__tn", "__fp", "__fn", '__n')

    @environment.torch.no_grad()
    def __init__(self):
        self.reset()

    @environment.torch.no_grad()
    def reset(self):
        """
        Reset internal attributes to calculate metric form batch 0 again.

        Returns
        -------
        None.

        """
        self.__tp = 0
        self.__tn = 0
        self.__fp = 0
        self.__fn = 0
        self.__n = 0

    @environment.torch.no_grad()
    def update(
            self,
            predicted: environment.torch.Tensor,
            targets: environment.torch.Tensor,
    ):
        """
        Update the internal representation with the new batch predictions.

        Parameters
        ----------
        outputs : environment.torch.Tensor
            DESCRIPTION.
        targets : environment.torch.Tensor
            DESCRIPTION.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        predicted = predicted.detach().int()
        targets = targets.detach().int()

        self.__tp += int(environment.torch.sum(
            (predicted == targets) & (predicted == 1)
        ))
        self.__tn += int(environment.torch.sum(
            (predicted == targets) & (predicted == 0)
        ))
        self.__fp += int(environment.torch.sum(
            (predicted != targets) & (predicted == 1)
        ))
        self.__fn += int(environment.torch.sum(
            (predicted != targets) & (predicted == 0)
        ))

        self.__n = self.__tp + self.__tn + self.__fp + self.__fn

    @metric_method
    @environment.torch.no_grad()
    def n(self) -> int:
        return self.__tp + self.__tn + self.__fp + self.__fn

    @metric_method
    @environment.torch.no_grad()
    def false_negative(self) -> int:
        return self.__fn

    @metric_method
    @environment.torch.no_grad()
    def false_positive(self) -> int:
        return self.__fp

    @metric_method
    @environment.torch.no_grad()
    def true_negative(self) -> int:
        return self.__tn

    @metric_method
    @environment.torch.no_grad()
    def true_positive(self) -> int:
        return self.__tp

    @metric_method
    @environment.torch.no_grad()
    def accuracy(self) -> float:
        return utils.safe_division(self.__tp + self.__tn, self.__n)

    @metric_method
    @environment.torch.no_grad()
    def balanced_accuracy(self) -> float:
        return utils.safe_division(
            self.true_positive_rate() + self.true_negative_rate(),
            2,
        )

    @metric_method
    @environment.torch.no_grad()
    def bookmaker_informedness(self) -> float:
        return self.informedness()

    @metric_method
    @environment.torch.no_grad()
    def critical_success_index(self) -> float:
        return self.threat_score()

    @metric_method
    @environment.torch.no_grad()
    def delta_p(self) -> float:
        return self.markedness()

    @metric_method
    @environment.torch.no_grad()
    def diagnostic_odds_ratio(self) -> float:
        return utils.safe_division(
            self.positive_likelihood_ratio(),
            self.negative_likelihood_ratio(),
        )

    @metric_method
    @environment.torch.no_grad()
    def fall_out(self) -> float:
        return self.false_discovery_rate()

    @metric_method
    @environment.torch.no_grad()
    def false_discovery_rate(self) -> float:
        return utils.safe_division(self.__fp, self.__tp + self.__fp)

    @metric_method
    @environment.torch.no_grad()
    def false_negative_rate(self) -> float:  # miss rate
        return utils.safe_division(self.__fn, self.__tp + self.__fn)

    @metric_method
    @environment.torch.no_grad()
    def false_omission_rate(self) -> float:
        return utils.safe_division(self.__fn, self.__tn + self.__fn)

    @metric_method
    @environment.torch.no_grad()
    def false_positive_rate(self) -> float:
        return utils.safe_division(self.__fp, self.__fp + self.__tn)

    @metric_method
    @environment.torch.no_grad()
    def f1_score(self) -> float:
        return utils.safe_division(
            2 * self.__tp,
            2 * self.__tp + self.__fp + self.__fn,
        )

    @metric_method
    @environment.torch.no_grad()
    def fowlkes_mallows_index(self) -> float:
        return (
            self.positive_predictive_value() * self.true_positive_rate()
        ) ** (
            1 / 2
        )

    @metric_method
    @environment.torch.no_grad()
    def hit_rate(self) -> float:
        return self.true_positive_rate()

    @metric_method
    @environment.torch.no_grad()
    def informedness(self) -> float:
        return self.true_positive_rate() + self.true_negative_rate() - 1

    @metric_method
    @environment.torch.no_grad()
    def jaccard_index(self) -> float:
        return self.threat_score()

    @metric_method
    @environment.torch.no_grad()
    def matthews_correlation_coefficient(self) -> float:
        return utils.safe_division(
            (
                self.__tp * self.__tn
            ) - (
                self.__fp * self.__fn
            ),
            (
                (
                    self.__tp + self.__fp
                ) * (
                    self.__tp + self.__fn
                ) * (
                    self.__tn + self.__fp
                ) * (
                    self.__tn + self.__fn
                )
            ) ** (
                1 / 2
            ),
        )

    @metric_method
    @environment.torch.no_grad()
    def markedness(self) -> float:
        return (
            self.positive_predictive_value() + self.negative_predictive_value() - 1
        )

    @metric_method
    @environment.torch.no_grad()
    def miss_rate(self) -> float:
        return self.false_negative_rate()

    @metric_method
    @environment.torch.no_grad()
    def negative_likelihood_ratio(self) -> float:
        return utils.safe_division(
            self.false_negative_rate(),
            self.true_negative_rate(),
        )

    @metric_method
    @environment.torch.no_grad()
    def negative_predictive_value(self) -> float:
        return utils.safe_division(self.__tn, self.__tn + self.__fn)

    @metric_method
    @environment.torch.no_grad()
    def positive_likelihood_ratio(self) -> float:
        return utils.safe_division(
            self.true_positive_rate(),
            self.false_positive_rate(),
        )

    @metric_method
    @environment.torch.no_grad()
    def positive_predictive_value(self) -> float:
        return utils.safe_division(self.__tp, self.__tp + self.__fp)

    @metric_method
    @environment.torch.no_grad()
    def precision(self) -> float:
        return self.positive_predictive_value()

    @metric_method
    @environment.torch.no_grad()
    def prevalence(self) -> float:
        return utils.safe_division(self.__tp + self.__fn, self.__n)

    @metric_method
    @environment.torch.no_grad()
    def prevalence_threshold(self) -> float:
        tpr = self.true_positive_rate()
        fpr = self.false_positive_rate()
        return utils.safe_division(
            (tpr * fpr) ** (1 / 2) - fpr,
            tpr - fpr,
        )

    @metric_method
    @environment.torch.no_grad()
    def probability_of_detection(self) -> float:
        return self.true_positive_rate()

    @metric_method
    @environment.torch.no_grad()
    def probability_of_false_alarm(self) -> float:
        return self.false_positive_rate()

    @metric_method
    @environment.torch.no_grad()
    def recall(self) -> float:
        return self.true_positive_rate()

    @metric_method
    def selectivity(self) -> float:
        return self.true_negative_rate()

    @metric_method
    def sensitivity(self) -> float:
        return self.true_positive_rate()

    @metric_method
    def specificity(self) -> float:
        return self.true_negative_rate()

    @metric_method
    def threat_score(self) -> float:
        return utils.safe_division(
            self.__tp,
            self.__tp + self.__fn + self.__fp,
        )

    @metric_method
    def true_negative_rate(self) -> float:
        return utils.safe_division(self.__tn, self.__fp + self.__tn)

    @metric_method
    def true_positive_rate(self) -> float:   # recall / sensitivity
        return utils.safe_division(self.__tp, self.__tp + self.__fn)

    # def __repr__(self) -> str:
    #     m = ", ".join(f"{k}={v:.4f}" for k, v in self.summary().items())
    #     return f"{self.__class__.__name__}({m})"


class MultiLabelMetrics(MetricsClass):
    __slots__ = ('__number_classes', '__weights',
                 '_tp_cls', '_fp_cls', '_fn_cls', '_tn_cls', '_cm_ll')

    def __init__(self, number_classes: int, weights: environment.torch.Tensor):
        if weights.shape[0] != number_classes:
            raise ValueError("weights length must equal number_classes")

        self.__number_classes = number_classes
        self.__weights = weights.to(environment.TORCH_DEVICE)

        # contadores por clase
        self._tp_cls = environment.torch.zeros(number_classes, dtype=environment.torch.int64,
                                               device=environment.TORCH_DEVICE)
        self._fp_cls = self._tp_cls.clone()
        self._fn_cls = self._tp_cls.clone()
        self._tn_cls = self._tp_cls.clone()
        self._cm_ll = environment.torch.zeros(
            (number_classes, number_classes),
            dtype=environment.torch.int64,
            device=environment.TORCH_DEVICE)

    @environment.torch.no_grad()
    def reset(self):
        """
        Reset all metrics and true label counts.

        Returns
        -------
        None.

        """
        self._tp_cls.zero_()
        self._fp_cls.zero_()
        self._fn_cls.zero_()
        self._tn_cls.zero_()
        self._cm_ll.zero_()

    @environment.torch.no_grad()
    def update(
            self,
            predicted: environment.torch.Tensor,
            targets: environment.torch.Tensor,):
        """
        Update the internal representation with the new batch predictions.

        Parameters
        ----------
        predicted : environment.torch.Tensor
            DESCRIPTION.
        targets : environment.torch.Tensor
            DESCRIPTION.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        predicted = predicted.int()
        targets = targets.int()

        self._tp_cls += ((predicted == 1) & (targets == 1)).sum(0)
        self._fp_cls += ((predicted == 1) & (targets == 0)).sum(0)
        self._fn_cls += ((predicted == 0) & (targets == 1)).sum(0)
        self._tn_cls += ((predicted == 0) & (targets == 0)).sum(0)
        self._cm_ll += (targets.T.float() @ predicted.float()).long()

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _div(num: environment.torch.Tensor,
             den: environment.torch.Tensor) -> environment.torch.Tensor:
        """Evita división por 0 manteniendo la forma del tensor."""
        eps = 1e-12
        return num / (den + eps)

    def _weighted_ratio(self,
                        num: environment.torch.Tensor,
                        den: environment.torch.Tensor) -> float:
        """
        Media ponderada (por self.__weights) de un cociente por-clase.
        Devuelve escalar float.
        """
        ratio_per_class = self._div(num.float(), den.float())          # (L,)
        w_sum = float(self.__weights.sum()) + 1e-12
        return float((ratio_per_class * self.__weights).sum() / w_sum)

    # -------- Confusion matrices  ------------------------------------- #
    @metric_method
    @environment.torch.no_grad()
    def confusion_matrix_2x2(self) -> environment.torch.Tensor:
        """Global 2×2."""
        tp = int(self._tp_cls.sum())
        fp = int(self._fp_cls.sum())
        fn = int(self._fn_cls.sum())
        tn = int(self._tn_cls.sum())
        return environment.torch.tensor([[tp, fp],
                                         [fn, tn]])

    @metric_method
    @environment.torch.no_grad()
    def confusion_matrix_ll(self) -> environment.torch.Tensor:
        """L × L completa acumulada en la época."""
        return self._cm_ll.clone()

    # -----------------  métricas globales -----------------------------

    @metric_method
    @environment.torch.no_grad()
    def true_positive(self) -> float: return float(self._tp_cls.sum())
    @metric_method
    @environment.torch.no_grad()
    def false_positive(self) -> float: return float(self._fp_cls.sum())
    @metric_method
    @environment.torch.no_grad()
    def false_negative(self) -> float: return float(self._fn_cls.sum())
    @metric_method
    @environment.torch.no_grad()
    def true_negative(self) -> float: return float(self._tn_cls.sum())

    # -------------  métricas ponderadas por clase ---------------------
    @metric_method
    @environment.torch.no_grad()
    def accuracy(self) -> float:
        num = self._tp_cls + self._tn_cls
        den = num + self._fp_cls + self._fn_cls
        return self._weighted_ratio(num, den)

    @metric_method
    @environment.torch.no_grad()
    def precision(self) -> float:
        return self._weighted_ratio(self._tp_cls,
                                    self._tp_cls + self._fp_cls)

    @metric_method
    @environment.torch.no_grad()
    def recall(self) -> float:
        return self._weighted_ratio(self._tp_cls,
                                    self._tp_cls + self._fn_cls)

    hit_rate = probability_of_detection = recall

    @metric_method
    @environment.torch.no_grad()
    def specificity(self) -> float:
        return self._weighted_ratio(self._tn_cls,
                                    self._tn_cls + self._fp_cls)

    @metric_method
    @environment.torch.no_grad()
    def false_positive_rate(self) -> float:            # = Fall-out
        return self._weighted_ratio(self._fp_cls,
                                    self._fp_cls + self._tn_cls)

    fall_out = probability_of_false_alarm = false_positive_rate

    @metric_method
    @environment.torch.no_grad()
    def false_discovery_rate(self) -> float:          # FP / (TP + FP)
        return self._weighted_ratio(self._fp_cls,
                                    self._tp_cls + self._fp_cls)

    false_discovery = false_discovery_rate
    positive_predictive_value = precision

    @metric_method
    @environment.torch.no_grad()
    def false_negative_rate(self) -> float:            # = Miss rate
        return self._weighted_ratio(self._fn_cls,
                                    self._tp_cls + self._fn_cls)

    miss_rate = false_negative_rate

    @metric_method
    @environment.torch.no_grad()
    def negative_predictive_value(self) -> float:
        return self._weighted_ratio(self._tn_cls,
                                    self._tn_cls + self._fn_cls)

    @metric_method
    @environment.torch.no_grad()
    def false_omission_rate(self) -> float:
        return 1.0 - self.negative_predictive_value()

    @metric_method
    @environment.torch.no_grad()
    def f1_score(self) -> float:
        p = self.precision()
        r = self.recall()
        return utils.safe_division(2 * p * r, p + r)

    @metric_method
    @environment.torch.no_grad()
    def balanced_accuracy(self) -> float:
        return 0.5 * (self.recall() + self.specificity())

    # ------------------------------------------------------------------
    #  Métricas que combinan varias de las anteriores
    # ------------------------------------------------------------------
    @metric_method
    @environment.torch.no_grad()
    def positive_likelihood_ratio(self) -> float:
        return utils.safe_division(self.recall(),
                                   self.false_positive_rate())

    @metric_method
    @environment.torch.no_grad()
    def negative_likelihood_ratio(self) -> float:
        return utils.safe_division(self.false_negative_rate(),
                                   self.specificity())

    @metric_method
    @environment.torch.no_grad()
    def diagnostic_odds_ratio(self) -> float:
        return utils.safe_division(self.positive_likelihood_ratio(),
                                   self.negative_likelihood_ratio())

    @metric_method
    @environment.torch.no_grad()
    def critical_success_index(self) -> float:         # = Threat score
        return self._weighted_ratio(self._tp_cls,
                                    self._tp_cls + self._fp_cls + self._fn_cls)

    threat_score = critical_success_index              # alias

    @metric_method
    @environment.torch.no_grad()
    def informedness(self) -> float:                   # = TPR + TNR – 1
        return self.recall() + self.specificity() - 1.0

    @metric_method
    @environment.torch.no_grad()
    def markedness(self) -> float:                     # = PPV + NPV – 1
        return self.precision() + self.negative_predictive_value() - 1.0

    @metric_method
    @environment.torch.no_grad()
    def prevalence(self) -> float:
        tp_fn = self._tp_cls + self._fn_cls
        total = tp_fn + self._tn_cls + self._fp_cls
        return self._weighted_ratio(tp_fn, total)

    @metric_method
    @environment.torch.no_grad()
    def prevalence_threshold(self) -> float:
        tpr = self.recall()
        fpr = self.false_positive_rate()
        return utils.safe_division((tpr * fpr) ** 0.5 - fpr, tpr - fpr)

    @metric_method
    @environment.torch.no_grad()
    def matthews_correlation_coefficient(self) -> float:
        tp = float(self._tp_cls.sum())
        fp = float(self._fp_cls.sum())
        fn = float(self._fn_cls.sum())
        tn = float(self._tn_cls.sum())
        num = tp * tn - fp * fn
        den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        return utils.safe_division(num, den)


AVAILABLE_METRICS_CLASSES = {
    'binary_class_metrics': BinaryClassMetrics,
    'multi_label_metrics': MultiLabelMetrics,
}
