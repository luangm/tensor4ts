import PairwiseOp from "../op/pairwise/PairwiseOp";
import Operation from "../op/Operation";
import TransformOp from "../op/transform/TransformOp";
import ReductionOp from "../op/reduction/ReductionOp";
import IndexOp from "../op/index/IndexOp";
import ComparisonOp from "../op/comparison/ComparisonOp";
import PairwiseExecutor from "./PairwiseExecutor";
import ReductionExecutor from "./ReductionExecutor";
import TransformExecutor from "./TransformExecutor";
import IndexExecutor from "./IndexExecutor";
import ComparisonExecutor from "./ComparisonExecutor";
import ConditionalOp from "../op/ternary/ConditionalOp";
import ConditionalExecutor from "./ConditionalExecutor";
import NaryOp from "../op/nary/NaryOp";
import NaryExecutor from "./NaryExecutor";

/**
 * Executor class is used to execute Ops
 * The executor implementation may be changed to use multiple threads / workers
 * An parallel optimization for execution could be split the inputs into multiple sub tensors and let worker run on each.
 */
export class Executor {

  /**
   * Runs an op. Does NOT return.
   * The caller is expected to grab result from op.result
   *
   * This function loops through the Tensor with consideration of buffer index
   */
  exec(op: Operation): void {

    if (op.isSpecial) {
      op.exec();
      return;
    }

    if (op instanceof ConditionalOp) {
      ConditionalExecutor.exec(op);
      return;
    }

    if (op instanceof ComparisonOp) {
      ComparisonExecutor.exec(op);
      return;
    }

    if (op instanceof PairwiseOp) {
      PairwiseExecutor.exec(op);
      return;
    }

    if (op instanceof TransformOp) {
      TransformExecutor.exec(op);
      return;
    }

    if (op instanceof ReductionOp) {
      ReductionExecutor.exec(op);
      return;
    }

    if (op instanceof IndexOp) {
      IndexExecutor.exec(op);
      return;
    }

    if (op instanceof NaryOp) {
      NaryExecutor.exec(op);
      return;
    }

    throw new Error("Cannot Execute Unknown Op");
  }
}

export default new Executor();