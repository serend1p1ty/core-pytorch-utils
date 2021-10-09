import sys

sys.path.append(".")
import logging

from cpu.logger import setup_logger

print("----------Test case 1----------")
parent_logger = setup_logger("parent_logger")
parent_logger.debug("debug")
parent_logger.info("info")
parent_logger.warning("warning")
parent_logger.error("error")
parent_logger.critical("critical")

child_logger1 = logging.getLogger("parent_logger.child_logger1")
child_logger1.debug("debug")
child_logger1.info("info")
child_logger1.warning("warning")
child_logger1.error("error")
child_logger1.critical("critical")

child_logger2 = logging.getLogger("parent_logger.child_logger2")
child_logger2.debug("debug")
child_logger2.info("info")
child_logger2.warning("warning")
child_logger2.error("error")
child_logger2.critical("critical")

# print("----------Test case 2----------")
# setup_logger()

# parent_logger = logging.getLogger("parent_logger")
# parent_logger.debug("debug")
# parent_logger.info("info")
# parent_logger.warning("warning")
# parent_logger.error("error")
# parent_logger.critical("critical")

# child_logger1 = logging.getLogger("parent_logger.child_logger1")
# child_logger1.debug("debug")
# child_logger1.info("info")
# child_logger1.warning("warning")
# child_logger1.error("error")
# child_logger1.critical("critical")

# child_logger2 = logging.getLogger("parent_logger.child_logger2")
# child_logger2.debug("debug")
# child_logger2.info("info")
# child_logger2.warning("warning")
# child_logger2.error("error")
# child_logger2.critical("critical")

# print("----------Test case 3----------")
# parent_logger = setup_logger("parent_logger")
# parent_logger.debug("debug")
# parent_logger.info("info")
# parent_logger.warning("warning")
# parent_logger.error("error")
# parent_logger.critical("critical")

# child_logger1 = setup_logger("parent_logger.child_logger1")
# child_logger1.debug("debug")
# child_logger1.info("info")
# child_logger1.warning("warning")
# child_logger1.error("error")
# child_logger1.critical("critical")

# child_logger2 = setup_logger("parent_logger.child_logger2")
# child_logger2.debug("debug")
# child_logger2.info("info")
# child_logger2.warning("warning")
# child_logger2.error("error")
# child_logger2.critical("critical")
