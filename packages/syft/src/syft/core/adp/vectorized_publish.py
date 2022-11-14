# future
from __future__ import annotations

# stdlib
from collections.abc import Iterable
from copy import deepcopy
import secrets
from typing import Callable
from typing import TYPE_CHECKING
from typing import Tuple

# third party
import jax
from jax import numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

# relative
from ...core.node.common.node_manager.user_manager import RefreshBudgetException
from ...core.tensor.autodp.gamma_tensor_ops import GAMMA_TENSOR_OP
from ..tensor.fixed_precision_tensor import FixedPrecisionTensor
from ..tensor.lazy_repeat_array import lazyrepeatarray
from ..tensor.passthrough import PassthroughTensor  # type: ignore
from .data_subject_ledger import DataSubjectLedger
from .data_subject_ledger import RDPParams
from .data_subject_ledger import compute_rdp_constant
from .data_subject_list import DataSubjectList

if TYPE_CHECKING:
    # relative
    from ..tensor.autodp.gamma_tensor import GammaTensor


@jax.jit
def calculate_bounds_for_mechanism(
    value_array: jnp.ndarray,
    min_val_array: jnp.ndarray,
    max_val_array: jnp.ndarray,
    sigma: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.array, jnp.array]:
    ones_like = jnp.ones_like(value_array)
    one_dim = jnp.reshape(ones_like, -1)

    worst_case_l2_norm = (
        jnp.sqrt(jnp.sum(jnp.square(max_val_array - min_val_array))) * one_dim
    )

    l2_norm = jnp.sqrt(jnp.sum(jnp.square(value_array))) * one_dim
    return l2_norm, worst_case_l2_norm, one_dim * sigma, one_dim


def publish(
    tensor: GammaTensor,
    ledger: DataSubjectLedger,
    get_budget_for_user: Callable,
    deduct_epsilon_for_user: Callable,
    sigma: float,
    is_linear: bool = True,
    private: bool = True,
) -> np.ndarray:
    print("\n\n===>>> vectorize publish", "tensor", tensor)
    print("\n\n===>>> vectorize publish", "ledger", ledger)
    print("\n\n===>>> vectorize publish", "sigma", sigma)
    print("\n\n===>>> vectorize publish", "is_linear", is_linear)
    print("\n\n===>>> vectorize publish", "private", private)
    """
    This method applies Individual Differential Privacy (IDP) as defined in
    https://arxiv.org/abs/2008.11193
        - Key results: Theorem 2.7 and 2.8 show how much privacy budget is spent by a query.

    Given a tensor, it checks if the user (a data scientist) has enough privacy budget (PB)
    to see data from every data subject.
    - If the user has enough privacy budget, then DP noise is directly added to the result,
      and PB is deducted.
    - If the user doesn't have enough PB, then every data subject with a higher epsilon
      than their PB's data is removed.
        - The epsilons are then recomputed to see if the user now has enough PB to see the
          remaining data or not.

    Notes:
        - The privacy budget spent by a query equals the maximum epsilon increase of any
          data subject in the dataset.
    """
    # Step 0: Ensure our Tensor's private data is in a form that is usable.
    if isinstance(tensor.child, FixedPrecisionTensor):
        # Incase SMPC is involved, there will be an FPT in the chain to account for
        value = tensor.child.decode()
    else:
        value = tensor.child

    while isinstance(value, PassthroughTensor):
        # TODO: Ask Rasswanth why this check is necessary
        # ATTENTION: we do the same unboxing below with root_child
        # is this still needed to be done twice?
        value = value.child

    # Step 1: We obtain all the parameters needed to calculate Epsilons
    if isinstance(tensor.min_vals, lazyrepeatarray):
        min_val_array = tensor.min_vals.to_numpy()
    else:
        min_val_array = tensor.min_vals

    if isinstance(tensor.max_vals, lazyrepeatarray):
        max_val_array = tensor.max_vals.to_numpy()
    else:
        max_val_array = tensor.max_vals

    if isinstance(tensor.data_subjects, np.ndarray):
        root_child = None
        while isinstance(tensor, PassthroughTensor):
            root_child = tensor.child
            tensor = root_child
        input_entities = tensor.data_subjects
    elif isinstance(tensor.data_subjects, DataSubjectList):
        input_entities = tensor.data_subjects.data_subjects_indexed
    else:
        raise NotImplementedError(
            f"Undefined behaviour for data subjects type: {type(tensor.data_subjects)}"
        )

    print("\n\n===>>> calculate_bounds_for_mechanism", "value", value)
    print("\n\n===>>> calculate_bounds_for_mechanism", "min_val_array", min_val_array)
    print("\n\n===>>> calculate_bounds_for_mechanism", "max_val_array", max_val_array)
    print("\n\n===>>> calculate_bounds_for_mechanism", "sigma", sigma)
    l2_norms, l2_norm_bounds, sigmas, coeffs = calculate_bounds_for_mechanism(
        value_array=value,
        min_val_array=min_val_array,
        max_val_array=max_val_array,
        sigma=sigma,
    )
    print("\n\n===>>> calculate_bounds_for_mechanism", "l2_norms", l2_norms)
    print("\n\n===>>> calculate_bounds_for_mechanism", "l2_norm_bounds", l2_norm_bounds)
    print("\n\n===>>> calculate_bounds_for_mechanism", "sigmas", sigmas)
    print("\n\n===>>> calculate_bounds_for_mechanism", "coeffs", coeffs)

    # its important that its the same type so that eq comparisons below dont break
    zeros_like = jnp.zeros_like(tensor.child)

    # this prevents us from running in an infinite loop
    previous_budget = None
    previous_spend = None

    # if we dont return below we will terminate if the tensor gets replaced with zeros
    prev_tensor = None

    while can_reduce_further(value=tensor.child, zeros_like=zeros_like):
        if prev_tensor is None:
            prev_tensor = tensor.child
        else:
            if (prev_tensor == tensor.child).all():  # type: ignore
                raise Exception("Tensor has not changed and is not all zeros")
            else:
                prev_tensor = tensor.child

        if is_linear:
            lipschitz_bounds = coeffs.copy()
        else:
            lipschitz_bounds = tensor.lipschitz_bound

        rdp_params = RDPParams(
            sigmas=sigmas,
            l2_norms=l2_norms,
            l2_norm_bounds=l2_norm_bounds,
            Ls=lipschitz_bounds,
            coeffs=coeffs,
        )

        # Step 2: Calculate the epsilon spend for this query

        # rdp_constant = all terms in Theorem. 2.7 or 2.8 of https://arxiv.org/abs/2008.11193 EXCEPT alpha
        print(
            "\n\n===>>> compute_rdp_constant",
            "rdp_params",
            rdp_params,
            "private",
            private,
        )
        rdp_constants = compute_rdp_constant(rdp_params, private=private)
        print("\n\n===>>> rdp_constants", rdp_constants)

        print("Rdp constants", rdp_constants)
        all_epsilons = ledger._get_epsilon_spend(
            rdp_constants
        )  # This is the epsilon spend for ALL data subjects
        if any(all_epsilons < 0):
            raise Exception(
                "Negative budget spend not allowed in PySyft for safety reasons."
                "Please contact the OpenMined support team for help."
            )

        epsilon_spend = max(
            all_epsilons
        )  # This is the epsilon spend for the QUERY, a single float.

        print("\n\n===>>> epsilon_spend", epsilon_spend)

        if not isinstance(epsilon_spend, float):
            epsilon_spend = float(epsilon_spend)

        if epsilon_spend < 0:
            raise Exception(
                "Negative budget spend not allowed in PySyft for safety reasons."
                "Please contact the OpenMined support team for help."
            )

        # Step 3: Check if the user has enough privacy budget for this query
        privacy_budget = get_budget_for_user(verify_key=ledger.user_key)
        has_budget = epsilon_spend <= privacy_budget

        # if we see the same budget and spend twice in a row we have failed to reduce it
        if (
            privacy_budget == previous_budget
            and epsilon_spend == previous_spend
            and not has_budget
        ):
            raise Exception(
                "Publish has failed to reduce spend. "
                f"With Budget: {previous_budget} Spend: {epsilon_spend}. Aborting."
            )

        if previous_budget is None:
            previous_budget = privacy_budget

        if previous_spend is None:
            previous_spend = epsilon_spend

        # Step 4: Path 1 - If the User has enough Privacy Budget, we just add noise,
        # deduct budget, and return the result.
        if has_budget:
            original_output = tensor.child

            # We sample noise from a cryptographically secure distribution
            # TODO: Replace with discrete gaussian distribution instead of regular
            # gaussian to eliminate floating pt vulns
            noise = np.asarray(
                [
                    secrets.SystemRandom().gauss(0, sigma)
                    for _ in range(original_output.size)
                ]
            ).reshape(original_output.shape)

            print("\n\n===>>> noise", noise)

            # The user spends their privacy budget before getting the result
            attempts = 0
            while attempts < 5:
                attempts += 1
                try:
                    ledger.spend_epsilon(
                        deduct_epsilon_for_user=deduct_epsilon_for_user,
                        epsilon_spend=epsilon_spend,
                        old_user_budget=privacy_budget,
                    )
                    break
                except RefreshBudgetException:  # nosec
                    ledger.spend_epsilon(
                        deduct_epsilon_for_user=deduct_epsilon_for_user,
                        epsilon_spend=epsilon_spend,
                        old_user_budget=privacy_budget,
                    )

                except Exception as e:
                    print(f"Problem spending epsilon. {e}")
                    raise e

            # The RDP constants are adjusted to account for the amount of exposure every
            # data subject's data has had.
            print(
                "\n\n===>>> ledger.update_rdp_constants", "rdp_constants", rdp_constants
            )
            print(
                "\n\n===>>> ledger.update_rdp_constants",
                "input_entities",
                input_entities,
            )
            ledger.update_rdp_constants(
                query_constants=rdp_constants, entity_ids_query=input_entities
            )
            ledger._write_ledger()
            print(
                "\n\n===>>> returning published results",
                "original_output",
                original_output,
            )
            print("\n\n===>>> returning published results", "noise", noise)
            print(
                "\n\n===>>> returning published results",
                "original_output + noise",
                original_output + noise,
            )
            final_result = original_output + noise
            print("\n\n===>>> final_result", type(final_result), final_result)

            return np.array(original_output + noise)

        # Step 4: Path 2 - User doesn't have enough privacy budget.
        elif not has_budget:
            print("Not enough privacy budget, about to start filtering.")
            # If the user doesn't have enough PB, they shouldn't see data of high
            # epsilon data subjects (privacy violation)
            # So we will remove data belonging to these data subjects from the computation.

            # Step 4.1: Figure out which data subjects are within the PB & the highest
            # possible spend
            # within_budget_filter = (
            #     jnp.ones_like(all_epsilons) * privacy_budget >= all_epsilons
            # )
            # highest_possible_spend = jnp.max(all_epsilons * within_budget_filter)
            # TODO: Modify to work with private/public operations (when input_tensor is a scalar)

            # Step 4.2: Figure out which Tensors in the Source dictionary have those data subjects

            # create a seperate iterable of the keys so they can be mutated below
            print("\n\n===>>> tensor", type(tensor), tensor)
            print("\n\n===>>> tensor.sources", type(tensor.sources), tensor.sources)
            print(
                "\n\n===>>> tensor.sources.sources",
                type(
                    getattr(tensor.sources, "sources", None),
                ),
                getattr(tensor.sources, "sources", None),
            )

            print("\n\n===>>> tensor.sources id", id(tensor.sources))
            filtered_sourcetree = deepcopy(tensor.sources)
            print("\n\n===>>> filtered_sourcetree", filtered_sourcetree)
            print("\n\n===>>> filtered_sourcetree id", id(filtered_sourcetree))
            input_tensors = list(filtered_sourcetree.values())
            print("\n\n===>>> input_tensors", input_tensors)

            parent_branch = [
                filtered_sourcetree for _ in input_tensors
            ]  # TODO: Ensure this isn't deepcopying!

            print("\n\n===>>> parent_branch", parent_branch)

            # relative
            from ..tensor.autodp.gamma_tensor import GammaTensor

            # ATTENTION: is this the same as tensor.sources.items() ?

            print(
                "\n\n===>>> zip(parent_branch, input_tensors)",
                zip(parent_branch, input_tensors),
            )

            for parent_state, input_tensor in zip(parent_branch, input_tensors):
                print("\n\n===>>> looping")
                print(
                    "\n\n===>>> parent_state, input_tensor", parent_state, input_tensor
                )
                print(
                    "\n\n===>>> len(parent_branch), len(input_tensors)",
                    len(parent_branch),
                    len(input_tensors),
                )

                if isinstance(input_tensor, GammaTensor):
                    if (
                        input_tensor.func_str == GAMMA_TENSOR_OP.NOOP.value
                    ):  # This is raw, unprocessed private data. Filter if eps spend > PB!
                        # Calculate epsilon spend for this tensor
                        print(
                            "\n\n===>>> input_tensor",
                            "input_tensor.func_str == GAMMA_TENSOR_OP.NOOP.value",
                        )
                        l2_norms = jnp.sqrt(jnp.sum(jnp.square(input_tensor.child)))
                        print("\n\n===>>> l2_norms", l2_norms)
                        rdp_params = RDPParams(
                            sigmas=sigmas,
                            l2_norms=l2_norms,
                            l2_norm_bounds=l2_norm_bounds,
                            Ls=lipschitz_bounds,
                            coeffs=coeffs,
                        )
                        print("\n\n===>>> rdp_params", rdp_params)

                        # Privacy loss associated with this private data specifically
                        epsilon = max(
                            ledger._get_epsilon_spend(
                                np.asarray(
                                    compute_rdp_constant(rdp_params, private=private)
                                )
                            )
                        )
                        print("\n\n===>>> epsilon", epsilon)

                        # Filter if > privacy budget
                        if jnp.isnan(epsilon):
                            raise Exception("Epsilon is NaN")

                        if epsilon > privacy_budget:
                            print(
                                "\n\n===>>> epsilon > privacy_budget",
                                epsilon > privacy_budget,
                            )
                            filtered_tensor = input_tensor.filtered()
                            print("\n\n===>>> filtered_tensor", filtered_tensor)

                            # Replace the original tensor with this filtered one
                            # remove the original state id

                            # TODO: This changes insertion order which could impact
                            # the non commutable operations like div where by the
                            # values are unpacked in order of insertion before being
                            # executed as a tuple
                            # see def _truediv(state: dict) -> jax.numpy.DeviceArray:
                            # in gamma_functions.py
                            print(
                                "\n\n===>>> parent_state keys before",
                                parent_state.keys(),
                            )
                            del parent_state[input_tensor.id]

                            # add the new zeroed state id tensor
                            parent_state[filtered_tensor.id] = filtered_tensor
                            print(
                                "\n\n===>>> parent_state keys after",
                                parent_state.keys(),
                            )
                        # If epsilon <= privacy budget, we don't need to do anything -
                        # the user has enough PB to use the data
                    else:
                        print(
                            "\n\n===>>> input_tensor",
                            "NOT input_tensor.func_str == GAMMA_TENSOR_OP.NOOP.value",
                        )
                        # Is this supposed to search the entire state tree with no_op
                        # data nodes being the leaves?

                        # ATTENTION: is this intended to walk the state tree?
                        # now that we no longer use recursion, I don't think this will
                        # work so we might want to have a stack / queue above to
                        # explore the frontier
                        print("\n\n===>>> input_tensors len before", len(input_tensors))
                        input_tensors += list(input_tensor.sources.values())
                        print("\n\n===>>> input_tensors len after", len(input_tensors))
                        print("\n\n===>>> parent_branch len before", len(parent_branch))
                        parent_branch += [
                            input_tensor.sources for _ in input_tensor.sources.values()
                        ]
                        print("\n\n===>>> parent_branch len after", len(parent_branch))
                else:
                    print("\n\n===>>> input_tensor", "NOT GAMMATENSOR")
                    # This is a public value, we don't touch 'em.
                    continue

            # Recompute the tensor's value now that some of its inputs have been
            # filtered, and repeat epsilon calculations

            # ATTENTION: When we swap state we need to handle the base case so that
            # the .child gets replaced not the .source (state) otherwise it never
            # terminates
            print("\n\n===>>> tensor before swap", tensor)
            tensor = tensor.swap_state(filtered_sourcetree)
            print("\n\n===>>> tensor after swap", tensor)
            print("tensor.child before restart: ", type(tensor.child), tensor.child)
            print("About to publish again with filtered source_tree!")

            # TODO: This isn't the most efficient way to do it since we can reuse sigmas, coeffs, etc.
            # TODO: Add a way to prevent infinite publishing?
            # TODO: Should we implement exponential backoff or something as a means of rate-limiting?
            # return new_tensor.publish(
            #     get_budget_for_user=get_budget_for_user,
            #     deduct_epsilon_for_user=deduct_epsilon_for_user,
            #     ledger=ledger,
            #     sigma=sigma,
            # )
        # Step 5: Revel in happiness.

        else:
            raise Exception

    print("\n\n===>>> returning absolute noise", "last tensor value was", tensor)
    noise = np.asarray(
        [secrets.SystemRandom().gauss(0, sigma) for _ in range(zeros_like.size)]
    ).reshape(zeros_like.shape)
    zeros = np.zeros_like(
        a=np.array([]), dtype=zeros_like.dtype, shape=zeros_like.shape
    )
    print("\n\n===>>> returning noise", "noise", noise)
    return zeros + noise


# each time we attempt to publish and filter values we need to ensure
# the while loop comparison is valid. We check for three edge cases.
# 1) the result of comparison is a non scalar because scalars cant be reduced
# 2) there are differences between the value and a zeros_like of the same shape
# 3) within the value ArrayLike there are no NaN values as these will never evaluate
# to True when compared with zeros_like and therefore never exit the loop
def can_reduce_further(value: ArrayLike, zeros_like: ArrayLike) -> bool:
    try:
        result = value != zeros_like
        # check we can call any or iterate on this value otherwise exit loop
        # numpy scalar types like np.bool_ are Iterable
        if not hasattr(result, "any") and not isinstance(result, Iterable):
            return False

        # make sure the comparison has some difference and there are also no NaNs
        # causing that difference in the result
        return result.any() and not_nans(result)
    except Exception as e:
        print(f"Unable to test reducability of {type(value)} and {type(zeros_like)}")
        raise e


# there are some types which numpy will not allow an isnan check such as strings
# so we should be extra careful to notify the user of why this check failed
def not_nans(value: ArrayLike) -> bool:
    try:
        # TODO: support nan ufunc properly?
        # do we need to add _array attrs to our tensor chain?
        while hasattr(value, "child"):
            value = value.child

        if isinstance(value, np.ndarray):
            return not np.isnan(value).any()
        else:
            return not jnp.isnan(value).any()
    except Exception as e:
        print(f"Holy Batmanananana. isnan is not supported {type(value)}.")
        raise e
